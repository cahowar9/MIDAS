## Import Block ##
import os
import gc
import logging
import shutil
import numpy as np
from copy import deepcopy
from pathlib import Path
import subprocess
from subprocess import STDOUT


## Functions ##
def get_results(parameters, filename, job_failed=False): #!TODO: implement pin power reconstruction.
    """
    Currently supports cycle length, F_q, F_dh, and max boron.
    
    Updated by Nicholas Rollins. 09/27/2024
    """
    ## Initialize logging for the present file
    logger = logging.getLogger("MIDAS_logger")
    
    ## Prepare container for results
    results_dict = {}
    for res in ["cycle_length", "pinpowerpeaking", "fdeltah", "max_boron"]:
        results_dict[res] = {}
        results_dict[res]['value'] = []
        
    if not job_failed:
        ## Read file for parsing
        with open(filename + ".parcs_dpl", "r") as ofile:
            filestr = ofile.read()
        
        ## Split file by section
        res_str = filestr.split('===============================================================================')
        res_str = res_str[1].split('_______________________________________________________________________________')
        res_str = res_str[0].split('\n')
        
        ## Parse raw values by timestep
        efpd_list = []; boron_list = []; keff_list = []; fq_list = []; fdh_list = []
        for i in range(2, len(res_str)-1):
            res_val=res_str[i].split()
            
            efpd_list.append(float(res_val[9]))
            boron_list.append(float(res_val[14]))
            keff_list.append(float(res_val[2]))
            fq_list.append(float(res_val[7]))
            fdh_list.append(float(res_val[6]))
        
        del filestr, res_str, res_val #unload file contents to clean up memory
        
        results_dict["cycle_length"]["value"] = calc_cycle_length(efpd_list,boron_list,keff_list)
        results_dict["pinpowerpeaking"]["value"] = max(fq_list)
        results_dict["fdeltah"]["value"] = max(fdh_list)
        results_dict["max_boron"]["value"] = max(boron_list)
        
        ## Correct Boron value if non-critical
        if results_dict["max_boron"]["value"] == 1800.0: #!TODO: initial guess should be a variable. can this be read from output file?
            new_max_boron = 0
            for i in range(len(boron_list)): #!TODO: I think this serves to line up boron_list with keff_list. Could be replaced by index()
                if boron_list[i]== 1800.0:
                    boron_worth = 10.0 #pcm/ppm
                    excess_rho = (keff_list[i] - 1.0)*10**5 #pcm; excess reactivity
                    excess_boron = excess_rho/boron_worth #ppm
                    max_boron_corrected = 1800.0 + excess_boron
                    if mboron > new_max_boron:
                        new_max_boron = mboron
            results_dict["max_boron"]["value"] = new_max_boron
    
    else: #job has failed; fill parameters with absurdly negative values.
        results_dict["cycle_length"]["value"] = 0.0
        results_dict["pinpowerpeaking"]["value"] = 10.0
        results_dict["fdeltah"]["value"] = 10.0
        results_dict["max_boron"]["value"] = 10000
    
    for param in parameters.keys():
        if param in results_dict:
            parameters[param]['value'] = results_dict[param]["value"]
        else:
            logger.warning(f"Parameter '{param}' not supported in PARCS342 results parsing.")
    
    return parameters

def evaluate(solution, input):
    """
    #!TODO: write docstring.
    
    Updated by Nicholas Rollins. 10/03/2024
    """
## Initialize logging for the present file
    logger = logging.getLogger("MIDAS_logger")
    
## Create and move to unique directory for PARCS execution
    cwd = Path(os.getcwd())
    indv_dir = cwd.joinpath(input.results_dir_name / Path(solution.name))
    if not indv_dir.exists():
        logger.debug(f"Creating new results directory: {indv_dir}")
        os.mkdir(indv_dir)
    logger.debug(f"Changing to new working directory: {indv_dir}")
    os.chdir(indv_dir)

## Prepare depletion file template #!TODO: can this file be dynamically generated instead of copied? #!TODO: is this necessary?
    if input.map_size == 'quarter':
        if input.num_assemblies == 193:
            shutil.copyfile('/home1/nkrollin/midas/MIDAS/samples/xslib/' + 'boc_exp_quart193_18.dep', 'boc_exp.dep') #!TODO: change this path to global variable
    else: #assume full geometry if not quarter-core
        if input.num_assemblies == 193:
            shutil.copyfile('/home1/nkrollin/midas/MIDAS/samples/xslib/' + 'boc_exp_full193.dep', 'boc_exp.dep')
        elif input.num_assemblies == 157:
            shutil.copyfile('/home1/nkrollin/midas/MIDAS/samples/xslib/' + 'boc_exp_full157.dep', 'boc_exp.dep')
    
## Prepare values for file writing
    list_unique_xs = np.concatenate([value if isinstance(value,list) else np.concatenate(list(value.values()))\
                                    for value in input.xs_list.values()])

    ## Fill loading pattern with chromosome
    fuel_locations = [loc for loc in input.core_dict.keys() if 2 < len(loc) <  5]
    soln_fuel_locations = {}
    for i in range(len(solution.chromosome)):
        soln_fuel_locations[fuel_locations[i]] = solution.chromosome[i]
    
    soln_core_dict = deepcopy(input.core_dict)
    for loc, label in soln_fuel_locations.items():
        tag = None
        for fueltype in input.tag_list['fuel']:
            if fueltype[1] == label:
                tag = fueltype[0]
        if not tag:
            raise ValueError("FA label not found in tag_list.")
        soln_core_dict[loc]['Value'] = tag
    #!for loc, label in soln_refl_locations.items(): #!TODO: create a way to specify reflector locs for multiple radial refls.

    soln_core_lattice = deepcopy(input.core_lattice) # core lattice filled with chromosome
    for loc, vals in soln_core_dict.items():
        sym_locs = [loc] + vals['Symmetric_Assemblies']
        for j in range(len(soln_core_lattice)):
            for i in range(len(soln_core_lattice[j])):
                if soln_core_lattice[j][i] in sym_locs:
                    if soln_core_lattice[j][i][0] == "R" and len(soln_core_lattice[j][i]) >= 5: #reflector
                        soln_core_lattice[j][i] = "10" #!TODO: add support more multiple radial refls.
                    else:
                        soln_core_lattice[j][i] = vals['Value']
    
## Generate Input File
    filename = solution.name + '.inp'
    
    ## CaseID Block ##
    with open(filename,"w") as ofile:
        ofile.write("!******************************************************************************\n")
        ofile.write('CASEID {}  \n'.format(solution.name))
        ofile.write("!******************************************************************************\n\n")

    ## CNTL Block ##
    with open(filename,"a") as ofile:
        ofile.write("CNTL\n")
        ofile.write("      RUN_OPTS   F T F F\n")
        if input.th_fdbk:
            ofile.write("      TH_FDBK    T\n")
            ofile.write("      INT_TH     T -1\n")
        else:
            ofile.write("      TH_FDBK    F\n")
        ofile.write("      CORE_POWER 100.0\n")
        ofile.write("      CORE_TYPE  PWR\n")
        ofile.write("      PPM        1000 1.0 1800.0 10.0\n")
        ofile.write("      DEPLETION  T  1.0E-5 T\n")
        ofile.write("      TREE_XS    T  {}  T  T  F  F  T  F  T  F  T  F  T  T  T  F  F \n".format(int(len(list_unique_xs))))
        ofile.write("      BANK_POS   100 100 100 100 100 100\n")
        ofile.write("      XE_SM      1 1 1 1\n")
        ofile.write("      SEARCH     PPM\n")
        ofile.write("      XS_EXTRAP  1.0 0.3\n")
        if input.pin_power_recon:
            ofile.write("      PIN_POWER  T\n")
        else:
            ofile.write("      PIN_POWER  F\n")
        ofile.write("      PRINT_OPT  T T T T T F T T T T  T  T  T  T  F  T  T")
        #!ofile.write("      PLOT_OPTS 0 0 0 0 0 2\n")
        ofile.write("\n")
        ofile.write("!******************************************************************************\n\n")
        
    ## PARAM Block ##
    with open(filename,"a") as ofile:
        ofile.write("PARAM\n")
        ofile.write("      LSOLVER     1 1 20\n")
        ofile.write("      NODAL_KERN  NEMMG\n")
        ofile.write("      CMFD        2\n")
        ofile.write("      DECUSP      2\n")
        ofile.write("      INIT_GUESS  0\n")
        ofile.write("      CONV_SS     1.e-6 5.e-5 1.e-3 0.001\n")
        ofile.write("      EPS_ERF     0.010\n")
        ofile.write("      EPS_ANM     0.000001\n")
        ofile.write("      NLUPD_SS    5 5 1\n")
        ofile.write("\n")
        ofile.write("!******************************************************************************\n\n")
    
    ## GEOM Block Inputs ##
    with open(filename,"a") as ofile:
        ofile.write("GEOM\n")
        if input.map_size == 'quarter':
            dim_size = [np.floor(input.nrow/2)+1, np.floor(input.ncol/2)+1]
        else: #assume full geometry if not quarter-core
            dim_size = [input.nrow, input.ncol]
        ofile.write(f"      GEO_DIM {dim_size[0]} {dim_size[1]} {input.number_axial} 1 1\n")
        ofile.write("      RAD_CONF\n\n")
        for x in range(soln_core_lattice.shape[0]):
            ofile.write("      ")
            for y in range(soln_core_lattice.shape[1]):
                ofile.write(soln_core_lattice[x,y])
                ofile.write("  ")
            ofile.write("\n")
        ofile.write("\n")
    
        assembly_width = 21.50 #!TODO: change this to an input with default.
        if input.map_size == 'quarter':
            ofile.write(f"      GRID_X      1*{assembly_width/2} {dim_size[0]-1}*{assembly_width}\n")
            ofile.write(f"      NEUTMESH_X  1*1 {dim_size[0]-1}*1\n")
            ofile.write(f"      GRID_Y      1*{assembly_width/2} {dim_size[0]-1}*{assembly_width}\n")
            ofile.write(f"      NEUTMESH_Y  1*1 {dim_size[0]-1}*1\n")
        else: #assume full geometry if not quarter-core
            ofile.write(f"      GRID_X      {dim_size[0]}*{assembly_width}\n")
            ofile.write(f"      NEUTMESH_X  {dim_size[0]}*1\n")
            ofile.write(f"      GRID_Y      {dim_size[1]}*{assembly_width}\n")
            ofile.write(f"      NEUTMESH_Y  {dim_size[1]}*1\n")
        ofile.write("      GRID_Z      {}\n".format('  '.join([str(x) for x in input.axial_nodes])))
        # Write radial reflectors
        xsnum_radtop = 2 + len(input.xs_list['reflectors']['radial'])
        rad_tags = [tag[0] for tag in input.tag_list['reflectors']]
        for i in range(len(input.xs_list['reflectors']['radial'])):
            tag = input.tag_list['reflectors'][rad_tags.index(input.tag_list['reflectors'][i][0])][0]
            ofile.write("      ASSY_TYPE   {}   1*1  {}*{}  1*{} REFL\n".format(tag,input.number_axial-2,2+i,xsnum_radtop))
        # Write fuel types
        if 'blankets' in input.fa_options:
            xsnum_fuel = xsnum_radtop + len(input.xs_list['blankets'])
        else:
            xsnum_fuel = xsnum_radtop
        for key in input.fa_options['fuel'].keys():
            fuel = input.fa_options['fuel'][key]
            xsnum_fuel += 1
            if 'blanket' in fuel:
                xsnum_blanket = xsnum_radtop + \
                                input.xs_list['blankets'].index(input.fa_options['blankets'][fuel['blanket']]['serial']) + 1
                ofile.write("      ASSY_TYPE   {}   1*1  1*{} {}*{}  1*{}  1*{} FUEL\n".format(fuel['type'],xsnum_blanket,\
                                                                                       input.number_axial-4,xsnum_fuel,\
                                                                                       xsnum_blanket,xsnum_radtop))
            else:
                ofile.write("      ASSY_TYPE   {}   1*1  {}*{}  1*{} FUEL\n".format(fuel['type'],input.number_axial-2,\
                                                                                  xsnum_fuel,xsnum_radtop))
        ofile.write("\n")

        if input.map_size == 'quarter':
            ofile.write("      BOUN_COND   0 2 0 2 2 2\n")
            ofile.write("      SYMMETRY 4\n")
        else: #assume full geometry if not quarter-core
            ofile.write("      BOUN_COND   2 2 2 2 2 2\n")
            ofile.write("      SYMMETRY 1\n")

        ofile.write("    PINCAL_LOC\n")
        for x in range(input.pincal_loc.shape[0]):
            ofile.write("      ")
            for y in range(input.pincal_loc.shape[1]):
                val = input.pincal_loc[x,y]
                try:
                    if not np.isnan(val):
                        ofile.write(str(input.pincal_loc[x,y]))
                        ofile.write("  ")
                except TypeError:
                    ofile.write(str(input.pincal_loc[x,y]))
                    ofile.write("  ")
            ofile.write("\n")
        ofile.write("\n")
        ofile.write("!******************************************************************************\n\n")

    ## FDBK Block ##
    with open(filename,"a") as ofile:
        ofile.write("FDBK\n")
        ofile.write("      FA_POWPIT     {} {}\n".format(np.round(input.power/input.num_assemblies,4),assembly_width))
        ofile.write("      GAMMA_FRAC    0.0208    0.0    0.0\n")
        ofile.write("      EFF_DOPLT   T  0.5556\n")
        ofile.write("\n")
        ofile.write("!******************************************************************************\n\n")

    ## TH Block ##
    with open(filename,"a") as ofile:
        ofile.write("TH\n")
        if input.th_fdbk:
            ofile.write("      FLU_TYP       0\n")
            ofile.write("      N_PINGT    264 25\n")
            ofile.write("      PIN_DIM      4.1 4.75 0.58 6.13\n")
            ofile.write("      FLOW_COND    {}  {}\n".format(np.round(input.inlet_temp-273.15,2),\
                                                             np.round(input.flow/input.num_assemblies,4)))
            ofile.write("      HGAP     11356.0\n") #!TODO:check this value, should it be parameterized?
            ofile.write("      N_RING   6\n")
            ofile.write("      THMESH_X       9*1\n")
            ofile.write("      THMESH_Y       9*1\n")
            ofile.write("      THMESH_Z       1 2 3 4 5 6 7 8 9 10 11 12\n")
        else:
            ofile.write("      UNIF_TH   0.740    626.85     {}\n".format(np.round(input.inlet_temp-273.15,2))) #!TODO: how to deal with av. fuel temp?
        ofile.write("\n")
        ofile.write("!******************************************************************************\n\n")

    ## DEPL Block ##
    with open(filename,"a") as ofile:
        ofile.write("DEPL\n")
        if input.calculation_type == 'single_cycle':
            ofile.write("      TIME_STP  1 1 4*30\n") #!TODO: parameterize this input.
        #!ofile.write("      INP_HST   './boc_exp.dep' -2 1\n") #!TODO: I don't believe this is necessary.
        ofile.write("      OUT_OPT   T  T  T  T  F\n")
        # Write reflector cross sections
        ofile.write("      PMAXS_F   1 '{}{}' 1\n".format(input.xs_lib / Path(input.xs_list['reflectors']['bot'][0]),\
                                                        input.xs_extension))
        for i in range(len(input.xs_list['reflectors']['radial'])):
            rxs_index = 2 + i
            radpath = input.xs_lib / Path(input.xs_list['reflectors']['radial'][i])
            ofile.write("      PMAXS_F   {} '{}{}' {}\n".format(rxs_index,radpath,input.xs_extension,rxs_index))
        ofile.write("      PMAXS_F   {} '{}{}' {}\n".format(rxs_index+1,\
                                                          input.xs_lib / Path(input.xs_list['reflectors']['top'][0]),\
                                                          input.xs_extension,rxs_index+1))
        nxs_index = rxs_index + 2
        # Write blankets cross sections
        if 'blankets' in input.fa_options:
            for i in range(len(input.xs_list['blankets'])):
                bxs_index = i + rxs_index + 2
                blanketpath = input.xs_lib / Path(input.xs_list['blankets'][i])
                ofile.write("      PMAXS_F   {} '{}{}' {}\n".format(bxs_index,blanketpath,input.xs_extension,bxs_index))
            nxs_index = bxs_index + 1
            
        # Write fuel types cross sections
        for i in range(len(input.xs_list['fuel'])):
            fxs_index = i + nxs_index
            ofile.write("      PMAXS_F   {} '{}{}' {}\n".format(fxs_index,\
                                                              input.xs_lib / Path(input.xs_list['fuel'][i]),\
                                                              input.xs_extension,fxs_index))
    
    ## MCYCL Block ##
    if input.calculation_type == 'eq_cycle':
        with open(filename,"a") as ofile:
            ofile.write("\n")
            ofile.write("!******************************************************************************\n\n")
            
            ofile.write("MCYCL\n")
            ofile.write("    CYCLE_DEF   1\n")
            ofile.write("      DEPL_STEP 1 1 17*30 18\n")
            ofile.write("      POWER_LEV 21*100.0\n")
            ofile.write("      BANK_SEQ  21*1\n\n")
            
            ofile.write("    LOCATION\n")
            for x in range(1,input.full_core_locs.shape[0]-1):
                for y in range(input.full_core_locs.shape[1]):
                    val = input.full_core_locs[x,y]
                    try:
                        if not np.isnan(val):
                            ofile.write(str(input.full_core_locs[x,y]))
                            ofile.write("  ")
                    except TypeError:
                        ofile.write(str(input.full_core_locs[x,y]))
                        ofile.write("  ")
                ofile.write("\n")
            ofile.write("\n")
            
            ofile.write("    SHUF_MAP   1   1\n")
            #!TODO: add shuffle map from chromosome.
            ofile.write("\n")
            
            ofile.write("    CYCLE_IND    1  0  1\n")
            for i in range(2,10): #!TODO: this max number of cycles could easily be a parameter.
                ofile.write(f"    CYCLE_IND    {i}  1  1\n")
            ofile.write(f"    CONV_EC    0.1  {i}\n")
    
    ## Terminate ##
    with open(filename,"a") as ofile:
        ofile.write(".")

## Run PARCS INPUT DECK #!TODO: separate the input writing and execution into two different functions that are called in sequence.
    parcscmd = "/cm1/codes/parcs_342/Executables/Linux/parcs-v342-linux2-intel-x64-release.x" #!TODO: move this to a global or environmental variable
    try:
        output = subprocess.check_output([parcscmd, filename], stderr=STDOUT, timeout=50) #wait until calculation finishes
    ## Get Results
        if 'Finished' in str(output): #job completed
            logging.debug(f"Job {solution.name} completed successfully.")
            ofile = solution.name + '.out'
            solution.parameters = get_results(solution.parameters, solution.name)
        else: #job failed
            logger.warning(f"Job {solution.name} has failed!")
            solution.parameters = get_results(solution.parameters, solution.name, job_failed=True)
    except subprocess.TimeoutExpired: #job timed out
        os.system('rm -f {}.parcs_pin*'.format(solution.name))
        logger.warning(f"Job {solution.name} has timed out!")
        solution.parameters = get_results(solution.parameters, solution.name, job_failed=True)
    
    logger.debug(f"Returning to original working directory: {cwd}")
    os.chdir(cwd)
    gc.collect()
    
    return solution

def calc_cycle_length(efpd,boron,keff):
    if boron[-1]==0.1:
        eoc1_ind = 0
        eco2_ind = len(efpd)
        for i in range(len(efpd)):
            if boron[i] > 0.1 and boron[i+1] == 0.1:
                eoc1_ind = i
                eco2_ind = i+1
        dbor = abs(boron[eoc1_ind-1]-boron[eoc1_ind])
        defpd = abs(efpd[eoc1_ind-1]-efpd[eoc1_ind])
        def_dbor = defpd/dbor
        eoc = efpd[eoc1_ind] + def_dbor*(boron[eoc1_ind]-0.1)
    elif boron[-1]==boron[0]==1800.0:
        drho_dcb=10 
        drho1 = (keff[-2]-1.0)*10**5
        dcb1 = drho1/drho_dcb
        cb1= boron[-2] + dcb1
        drho2 = (keff[-1]-1.0)*10**5
        dcb2 = drho2/drho_dcb
        cb2= boron[-1] + dcb2
        dbor = abs(cb1-cb2)
        defpd = abs(efpd[-2]-efpd[-1])
        def_dbor = defpd/dbor
        eoc = efpd[-1] + def_dbor*(cb2-0.1)
    else:
        dbor = abs(boron[-2]-boron[-1])
        defpd = abs(efpd[-2]-efpd[-1])
        def_dbor = defpd/dbor
        eoc = efpd[-1] + def_dbor*(boron[-1]-0.1)
    return eoc