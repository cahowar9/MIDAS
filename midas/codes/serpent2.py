import numpy as np
import logging
import subprocess
import os
import time
import re
from pathlib import Path
import math
from midas.utils import LWR_fuelcyclecost as fuelcyclecost
from scipy.interpolate import interp1d
import serpentTools as st
import matplotlib.pyplot as plt

## Initialize logging for the present file
logger = logging.getLogger("MIDAS_logger")

def evaluate(solution, input):

#Predefine exit codes for optional runs so later logic doesn't fail
    base_exit_code = None
    dop_exit_code = None
    dep_exit_code = None
    shutdown_exit_code = None
    results_dict = {}
    chromosome = solution.chromosome
    genome = input.genome
    template_dict = {}

    for key, value in sorted(genome.items(), key = lambda item: item[1]['index']):
        if 'continuous_range' in value:
            chromosome[value['index']] = float(chromosome[value['index']]*(value['continuous_range'][1]-value['continuous_range'][0]) + value['continuous_range'][0])
        elif 'discrete_range' in value:
            indices = np.where(np.isclose(value['normalized_discrete_range'], chromosome[value['index']], atol=1e-5))[0]
            chromosome[value['index']] = value['discrete_range'][indices[0]]
        template_dict[key] = chromosome[value['index']]

    # Convert normalized values back to real values and store them in a dictionary for easy access

    for key, value in sorted(genome.items(), key = lambda item: item[1]['index']):
        chromosome[value['index']] = float(chromosome[value['index']])
        template_dict[key] = chromosome[value['index']]
    if template_dict['fuel_pin_radius']> .5*template_dict['pin_pitch'] or template_dict['moderator_pin_radius']>.5*template_dict['pin_pitch']:
        template_dict['pin_pitch'] = 2*max(template_dict['fuel_pin_radius'], template_dict['moderator_pin_radius'])
        chromosome[0] = template_dict['pin_pitch']
        solution.chromosome[0] = ((template_dict['pin_pitch']))

    # Create and move to unique directory for Serpent execution

    cwd = Path(os.getcwd())
    indv_dir = Path(input.results_dir_name) / solution.name
    base_dir = indv_dir / "base"
    doppler_dir = indv_dir / "doppler"
    depletion_dir = indv_dir / "depletion"
    shutdown_dir = indv_dir / "shutdown"
    base_file = base_dir / "base_input"
    doppler_file = doppler_dir / "doppler_input"
    depletion_file = depletion_dir / "depletion_input"
    shutdown_file = shutdown_dir / "shutdown_input"
    if not indv_dir.exists():
        logger.debug(f"Creating new results directory: {indv_dir}")
        os.mkdir(indv_dir)
    logger.debug(f"Changing to new working directory: {indv_dir}")
    os.chdir(indv_dir)

    #Create subdirectories for base, DTC, shutdown, and depletion runs

    if not base_dir.exists():
        os.mkdir(base_dir)
    fill_template(input.input_template["loc"], base_file, template_dict)
    if not doppler_dir.exists() and "doppler_temperature_coefficient" in input.objectives:
        os.mkdir(doppler_dir)
    if not shutdown_dir.exists() and "keff_shutdown" in input.objectives:
        os.mkdir(shutdown_dir)
    if "doppler_temperature_coefficient" in input.objectives:
        fill_template(input.input_template["loc"], doppler_file, template_dict)
        remove_detector_lines(doppler_file,input.power_peaking_detectors)
        update_temp(doppler_file)
    if "keff_shutdown" in input.objectives:
        fill_template(input.shutdown_template, shutdown_file, template_dict)
    if input.depletion_settings['apply'] and not depletion_dir.exists():
        os.mkdir(depletion_dir)
    if input.depletion_settings['apply']:
        fill_template(input.input_template["loc"], depletion_file, template_dict)
        with open(depletion_file, "a") as f:
            f.write(f"\nset pop {input.depletion_settings['particles_per_cycle']} {input.depletion_settings['active_cycles']} {input.depletion_settings['inactive_cycles']}\n")
            if input.depletion_settings['depletion_units'].lower() == 'days':
                f.write("\ndep daytot\n")
            else:
                f.write("\ndep butot\n")
            for step in input.depletion_settings['depletion_steps']:
                f.write(f"{step}\n")
            f.close()
    #Remove detector lines outside of base case to improve speed
        remove_detector_lines(depletion_file,input.power_peaking_detectors)

#Add cross sections and population into serpent files
    for file in [base_file, doppler_file, depletion_file, shutdown_file]:
        if file.exists():
            with open(file, "a") as f:
                if file != depletion_file:
                    f.write(f"\nset pop {input.particles_per_cycle} {input.active_cycles} {input.inactive_cycles}\n")
                f.write(f'set acelib "{input.xs_lib}"\n')
                f.write(f'set declib "{input.dec_lib}"\n')
                f.write(f'set nfylib "{input.nfy_lib}"\n')
                f.close()

#Start depletion calc first since it takes the longest
    if input.depletion_settings['apply']:
        os.chdir(depletion_dir)
        make_sh_file(depletion_dir, "depletion_input", input.depletion_settings['omp_threads'])
        subprocess.run (["sbatch", "run.sh"])

#Unrealistically bad results to return if code fails to drive optimization away from this region
    fail_results = {'doppler_temperature_coefficient': 5, 'cycle_length': 0.01, 'cost_fuelcycle': 1000000000000,'total_mass': 1000000000,'fdeltah': 8,'max_doserate':50000,'keff':0.0001,'keff_shutdown':10}
#Run base calc and get results
    os.chdir(base_dir)
    make_sh_file(base_dir, "base_input", input.omp_threads)
    subprocess.run (["sbatch", "run.sh"])
    if doppler_dir.exists():
        os.chdir(doppler_dir)
        make_sh_file(doppler_dir, "doppler_input", input.omp_threads)
        subprocess.run (["sbatch", "run.sh"])
    if shutdown_dir.exists():
        os.chdir(shutdown_dir)
        make_sh_file(shutdown_dir, "shutdown_input", input.omp_threads)
        subprocess.run (["sbatch", "run.sh"])

    base_exit_code = check_out(base_dir)
    while base_exit_code == 1:
        base_exit_code = check_out(base_dir)
        time.sleep(60)
    if base_exit_code == 0:
        base_res_path = base_dir / "base_input_res.m"
        base_results = st.read(base_res_path)
        if "fdeltah" in input.objectives:
            base_det_path = base_dir / "base_input_det0.m"
            base_det_results = st.read(base_det_path)
            peaking_results = []
            for det in input.power_peaking_detectors: 
                if det in base_det_results.detectors:
                    peaking_results.append(max(base_det_results.detectors[det].tallies))
            mean_pow = np.mean(peaking_results)
            peaking_factors = peaking_results / mean_pow
            results_dict["fdeltah"] = np.max(peaking_factors)
    
        results_dict["keff"]=base_results.resdata["absKeff"][0]
        mass_dict = get_masses(base_dir / "base_input.out")

    #Exclude materials not to be included in mass calculation specified by user
        results_dict["total_mass"] = 0
        for key in mass_dict:
            if key in input.mass_materials or input.mass_materials == 'all':
            #Store mass of each included material and convert from g to lb
                results_dict["total_mass"] += float(mass_dict[key])/453.6

    if doppler_dir.exists():
        dop_exit_code = check_out(doppler_dir)
        while dop_exit_code == 1:
            dop_exit_code = check_out(doppler_dir)
            time.sleep(60)
        if dop_exit_code ==0 and base_exit_code == 0:
            dop_res_path = doppler_dir / "doppler_input_res.m"
            dop_results = st.read(dop_res_path)
            rho1 = (base_results.resdata["absKeff"][0] - 1)/base_results.resdata["absKeff"][0]
            rho2 = (dop_results.resdata["absKeff"][0] - 1)/dop_results.resdata["absKeff"][0] 
            results_dict["doppler_temperature_coefficient"] = ((rho2 - rho1) / 150) * 10**5

    if shutdown_dir.exists():
        shutdown_exit_code = check_out(shutdown_dir)
        while shutdown_exit_code == 1:
            shutdown_exit_code = check_out(shutdown_dir)
            time.sleep(60)
        if shutdown_exit_code == 0:
            shutdown_res_path = shutdown_dir / "shutdown_input_res.m"
            shutdown_results = st.read(shutdown_res_path)
            results_dict["keff_shutdown"] = shutdown_results.resdata["absKeff"][0]

    if depletion_dir.exists():
        dep_exit_code = check_out(depletion_dir)
        while dep_exit_code == 1:
            dep_exit_code = check_out(depletion_dir)
            time.sleep(60)
        if dep_exit_code == 0 and (dop_exit_code == 0 or dop_exit_code is None) and base_exit_code == 0 and (shutdown_exit_code == 0 or shutdown_exit_code is None):
            dep_path = depletion_dir / "depletion_input_res.m"
            dep_results = st.read(dep_path)
            burn_days = dep_results.resdata["burnDays"][:,0]
            burn_keff = dep_results.resdata["absKeff"][:,0] 
            interpolate = interp1d(burn_keff,burn_days,kind='linear',fill_value='extrapolate')
            if base_results.resdata["absKeff"][0] > 1.0:
                results_dict["cycle_length"] = interpolate(1.0)
            else:
                results_dict["cycle_length"] = 0.01 #If keff is already below 1, return very short cycle length to drive optimization away from this region
            
    if 'cost_fuelcycle' in input.objectives and dep_exit_code == 0 and (dop_exit_code == 0 or dop_exit_code is None) and base_exit_code == 0:
        cycle_cost = fuelcyclecost.calc_fuelcost_triso(template_dict["enrichment"]/100,(mass_dict['fuel']/453.6))
        results_dict['cost_fuelcycle'] = cycle_cost

    if (dep_exit_code != 0 and dep_exit_code is not None) or (dop_exit_code != 0 and dop_exit_code is not None) or base_exit_code != 0 or (shutdown_exit_code != 0 and shutdown_exit_code is not None):
        results_dict = fail_results

    for key, value in results_dict.items():
        if key in input.objectives:
            solution.parameters[key]["value"] = value
        else:
            logger.info(f"Objective {key} is available in serpent but not currently used by the optimization")

    return solution

def check_out(directory):
    # Regex pattern to match myopt.out.<numbers>
    pattern = re.compile(r'slurm-\d+\.out$')

    for filename in os.listdir(directory):
        if pattern.match(filename):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    line_lower = line.strip().lower()
                    if re.search(r'simulation aborted', line_lower):
                        return 2
                    if re.search(r'simulation complete', line_lower):
                        return 0
                    if re.search(r'serpent is finished for this midas run', line_lower):
                        return 2
    return 1


def fill_template(template_path, output_path, template_dict):
    """
    Fill a template file with values from template_dict and save to output_path.
    - Placeholders {var}, {var * 2}, {sin(var)}, etc. are supported.
    """

    # Safe math environment (only what you allow)
    safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    safe_env.update(template_dict)

    # Read template
    template_text = Path(template_path).read_text()

    # Regex: find everything inside {}
    pattern = re.compile(r"\{(.*?)\}")

    def replace_match(match):
        expr = match.group(1).strip()
        try:
            return str(eval(expr, {"__builtins__": {}}, safe_env))
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expr}': {e}")

    # Replace placeholders
    filled_text = pattern.sub(replace_match, template_text)

    # Save to output
    Path(output_path).write_text(filled_text)

def update_temp(filename):
    tmp_pattern = re.compile(r"(tmp\s+)([-+]?\d*\.?\d+)", re.IGNORECASE)

    with open(filename, "r") as f:
        lines = f.readlines()
        f.close()

    updated_lines = []
    for line in lines:
        if "mat" in line and "fuel" in line:  # quick filter
            # Replace tmp number with number+100
            def repl(match):
                prefix = match.group(1)  # "tmp "
                number = float(match.group(2))
                return f"{prefix}{number + 150:g}"  # keep clean formatting

            new_line = tmp_pattern.sub(repl, line)
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    with open(filename, "w") as f:
        f.writelines(updated_lines)
        f.close()

def get_masses(filepath):
    """
    Reads a Serpent output-like file and extracts the mass of all materials.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to read.

    Returns
    -------
    dict
        Dictionary mapping material names to masses in grams, e.g.,
        { "moderator": 1.13e6, "fuel": 8.7e5 }
    """
    filepath = Path(filepath)
    masses = {}
    current_material = None
    mass_pattern = re.compile(r"- Mass\s+([0-9.E+-]+)\s+g")

    with filepath.open("r") as f:
        for line in f:
            line_strip = line.strip()
            # Detect start of material block
            if line_strip.startswith("Material "):
                # Extract material name in quotes
                match_name = re.match(r'Material\s+"(.+?)"', line_strip)
                if match_name:
                    current_material = match_name.group(1)
                else:
                    current_material = None
                continue

            # If inside a material block, look for mass
            if current_material:
                match_mass = mass_pattern.search(line_strip)
                if match_mass:
                    masses[current_material] = float(match_mass.group(1))
                    current_material = None  # done with this material
        f.close()
    return masses

def remove_detector_lines(filepath, detector):
    """
    Remove lines from a file based on detector type.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to modify.
    detector : str or list of str
        If 'ppw', remove lines containing 'set adf' and 'set ppw'.
        If a list, remove lines containing 'det {detector[i]}' for each element.
    """
    filepath = Path(filepath)

    # Read all lines
    with filepath.open("r") as f:
        lines = f.readlines()
        f.close()

    # Determine lines to remove
    if detector == 'ppw':
        remove_keywords = ['set adf', 'set ppw']
    elif isinstance(detector, list):
        remove_keywords = [f"det {d}" for d in detector]

    # Filter lines
    new_lines = [line for line in lines if not any(keyword in line for keyword in remove_keywords)]

    # Write back
    with filepath.open("w") as f:
        f.writelines(new_lines)
        f.close()

def get_heavy_metal_percent(filepath):
    """
    Reads a Serpent material definition and calculates the weight percent
    of isotopes with Z = 92 (U) or 94 (Pu). Uses atomic masses from the 
    `periodictable` library when possible.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
        f.close()

    masses = {}
    mat_name = None
    fractions = {}

    # Regex for isotopes like 92235.06c
    iso_pattern = re.compile(r"^\s*(\d+)\.\d+\w*\s+(-?\d+\.?\d*([Ee][+-]?\d+)?)")

    for line in lines:
        if line.strip().startswith("mat") and "fuel" in line:
            mat_name = line.strip()
            continue

        match = iso_pattern.match(line)
        if match:
            zaid, frac, _ = match.groups()
            ZAI = int(zaid)
            Z = int(ZAI // 1000)     # first digits = Z
            A = int(ZAI % 1000)      # last 3 = mass number

            frac = float(frac)

            # Get atomic mass from mass number
            try:
                from periodictable import elements
                mass = getattr(elements, Z).isotopes[A].mass
            except Exception:
                mass = float(A)

            # Store
            fractions[(Z, A)] = (frac, mass)

    if not fractions:
        raise ValueError("No valid isotopes found in file.")

    # Compute total mass contribution
    total_mass = sum(abs(frac) for frac, _ in fractions.values())

    # Contribution from U & Pu only
    upu_mass = sum(abs(frac) for (Z, A), (frac, _) in fractions.items() if Z in [92, 94])

    # Weight percent
    weight_percent = (upu_mass / total_mass)
    return weight_percent

def make_sh_file(directory, input_file, omp_threads):
    """
    Create a SLURM shell script for running Serpent in the specified directory.

    Parameters
    ----------
    directory : str or Path
        Directory where the shell script will be created.
    omp_threads : int
        Number of OpenMP threads to use.
    """
    directory = Path(directory)
    sh_file_path = directory / "run.sh"
    text = "#!/bin/bash\n\n"
    text += '#SBATCH -J "midas_ss2"\n'
    text += "#SBATCH -p newq\n"
    text += "#SBATCH -t 1000:00:00\n"
    text += f"#SBATCH -n {omp_threads}\n"
    text += "#SBATCH -N 1\n"
    text += f"time sss2 -omp {omp_threads} {input_file}\n"
    text+= 'echo "Serpent is finished for this MIDAS run"\n'

    with sh_file_path.open("w") as f:
        f.write(text)
        f.close()
