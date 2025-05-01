## Import Block ##
import logging
from copy import deepcopy
import random
import numpy as np
from midas.utils import optimizer_tools as optools
import statistics

class Parallel_Simulated_Annealing():
    """
    Class for performing optimization using parallel simulated annealing.
    parallel simulated annealing works by running multiple SAs in parallel.
    It was decided to make a unique file for PSA to keep the code clean and organized.

    Note that within MIDAS each iterations is refered to as a generation. This is not necessarily the correct nomenclature 
    for SA but it is named in this way for consistencey.

    Written by Jake Mikouchi. 04/25/2025
    """

    def __init__(self, input):
        self.input = input   
        self.global_temperature = input.initial_temperature
        self.local_temperatures = [input.initial_temperature for i in range(self.input.num_procs)]
        self.buffer = []
        self.active_solutions = [[] for i in range(self.input.num_procs)]
        self.best_in_gen = [0 for i in range(self.input.num_procs)]
        self.total_moves = 0

    def reproduction(self, population_step, proc):
        """
        Generates a new individual by perturbing the origional solution. The perturbation 
        operates in a similar manner to mutations in genetic algorithm. 
        Specific for PSA
        
        Updated by Jake Mikouchi. 04/26/2025
        """
        if population_step == 0:
            # takes solution from buffer if begining of serial SA
            primary_individual = [PSA_reproduction.select_from_buffer(self, proc).chromosome]
            self.total_moves = 0

        else: 
            # if not beginning of algorithm then use stored active solution
            primary_individual = [PSA_reproduction.selection(self, proc).chromosome]

        individual_pairs = deepcopy(primary_individual)
        ## preserve core parameters 
        LWR_core_parameters = [self.input.nrow, self.input.ncol, self.input.num_assemblies, self.input.symmetry]
        ## Perform perturbation
        if self.input.perturbation_type == "perturb_by_gene":
            individual_pairs.append(PSA_reproduction.perturb_by_gene(self.input, primary_individual[0]))
        else:
            raise ValueError("Requested perturbation type not recognized.")

        self.active_solutions[proc] = individual_pairs

        self.local_temperatures[proc] = self.Temperature_update_methods(self.local_temperatures[proc], self.input.secondary_cooling_schedule, population_step, Global=False)

        return individual_pairs

    def update_active(self, full_active_solutions):
        """
        updates active_solutions with the final values

        Written by Jake Mikouchi. 04/29/2025
        """      
        self.active_solutions = full_active_solutions

    def update_buffer(self, pop_list, current_generation):
        """
        Updates solution buffer by filling it with solutions or replacing old solutions with new ones

        Written by Jake Mikouchi. 04/25/2025
        """

        if not self.buffer:
            for i in range(self.input.buffer_size):
                self.buffer.append(pop_list[i])

            self.global_temperature = Cooling_Schedule.lam_set_intial(self)
            logger = logging.getLogger("MIDAS_logger")
            logger.info(f"Initial Temperature: {self.global_temperature}")

        else: 
            potential_candidates = []
            for candidate in self.best_in_gen:
                if candidate not in self.buffer:
                    potential_candidates.append(candidate)

            for candidate in potential_candidates:
                # find smallest fitness value in buffer
                smallest_indv = self.buffer[0]
                for buf_indv in self.buffer:
                    if buf_indv.fitness_value < smallest_indv.fitness_value:
                        smallest_indv = buf_indv

                # replace smallest fitness value if candidate has a higher one
                if candidate.fitness_value > smallest_indv.fitness_value:
                    self.buffer.pop(self.buffer.index(smallest_indv))
                    self.buffer.append(candidate) 

            # calculate temperature using LAM cooling schedule
            logger = logging.getLogger("MIDAS_logger")
            self.global_temperature = self.Temperature_update_methods(self.global_temperature, self.input.cooling_schedule, current_generation, Global=True)
            logger.info(f"Updated Temperature: {self.global_temperature}")

        for i in range(len(self.local_temperatures)):
            self.local_temperatures[i] = self.global_temperature

    def Temperature_update_methods(self, temperature, cooling_schedule, current_step, Global):
        """
        Method for distributing to the requested SA cooling schedule

        updated by Jake Mikouchi. 04/25/2024
        """
        if Global:
            if cooling_schedule == 'exponential_decrease':
                temperature = Cooling_Schedule.exponential_decrease(temperature)
            if cooling_schedule == 'linear_update':
                temperature = Cooling_Schedule.linear_update(self.input.initial_temperature, current_step, self.input.num_generations)
            if cooling_schedule == 'log_update':
                temperature = Cooling_Schedule.logarithmic_update(self.input.initial_temperature, current_step)
            if cooling_schedule == 'lam':
                temperature = Cooling_Schedule.lam(self, current_step)  
        else: 
            if cooling_schedule == 'exponential_decrease':
                temperature = Cooling_Schedule.exponential_decrease(temperature)
            if cooling_schedule == 'linear_update':
                temperature = Cooling_Schedule.linear_update(self.global_temperature, current_step, self.input.population_size)
            if cooling_schedule == 'log_update':
                temperature = Cooling_Schedule.logarithmic_update(self.global_temperature, current_step)
 
        return temperature 
    

class PSA_reproduction():
    """
    Functions for performing reproduction of chromosomes using PSA
    methodologies.
     
    Written by Jake Mikouchi. 04/25/25
    """

    def select_from_buffer(self, proc):
        """
        selects a solution from the buffer to act as a starting point for the serial SAs
        """

        scalingfactor = 100
        # finds the sum of the probabilities of each individual being selected in the Buffer
        sumprob = 0
        for i in range(self.input.buffer_size):
            sumprob += np.exp((-1 * self.buffer[i].fitness_value) / scalingfactor)

        # finds the unique probability of each individual being selected in the Buffer
        positional_prob = []
        for i in range(self.input.buffer_size):
            currprob = np.exp((-1 * self.buffer[i].fitness_value) / scalingfactor)
            totalprob = (currprob / sumprob)
            positional_prob.append(totalprob)

       # uses the probabilities to select which indivdual to use as a solution
        sum_prob_list = []
        for i in range(self.input.buffer_size):
            if i >= 1:
                sum_prob_list.append(sum_prob_list[i - 1] + positional_prob[i])
            if i == 0:
                sum_prob_list.append(positional_prob[i])

        random_num = random.uniform(0, 1)

        for i in range(self.input.buffer_size):
            if i == 0:
                if random_num < sum_prob_list[i]:
                    selected_indv = self.buffer[i]
                    break
            if i >= 1:
                if random_num < sum_prob_list[i] and random_num >= sum_prob_list[i - 1]:
                    selected_indv  = self.buffer[i]
                    break
                    
        self.best_in_gen[proc] = selected_indv


        return selected_indv

    def selection(self, proc):
        """
        Selects the current indivdiual in the SA optimization.
        This is a little weird and may need to be addressed. MIDAS is constructed to always maximize the fitness
        but SA works by minimizing the cost. For now maintianing GA likeness is the priority so cost/fitness is maximized.
        
        Created by Jake Mikouchi. 04/22/2025
        """
        # optimizer.py does some weird shifting due to inactive solutions
        # so challenger is index 0 while primary is index 1
        challenger = self.active_solutions[proc][0]
        if len(self.active_solutions[proc]) < 2:
            primary = self.active_solutions[proc][0]
        else:
            primary = self.active_solutions[proc][1]

        selected = self.active_solutions[proc][0]

        if challenger.fitness_value >= primary.fitness_value:
            selected = challenger
        else: 
            acceptance_prob = np.exp(-1 * (primary.fitness_value - challenger.fitness_value) / self.local_temperatures[proc]) 
            chance = random.random()
            if chance < acceptance_prob:
                selected = challenger
                self.total_moves += 1
            else: 
                selected = primary

        if challenger.fitness_value >= self.best_in_gen[proc].fitness_value:
            self.best_in_gen[proc] = challenger

        return selected

## Mutation types ##
    def perturb_by_gene(input_obj, chromosome):
        """
        Generates a new solution by randomly mutating a single gene.
        
        Created by Jake Mikouchi. 04/21/2025
        """
        ## Initialize logging for the present file
        logger = logging.getLogger("MIDAS_logger")
        
        LWR_core_parameters = [input_obj.nrow, input_obj.ncol, input_obj.num_assemblies, input_obj.symmetry]
        
        if input_obj.calculation_type in ["eq_cycle"]:
            zone_chromosome = [loc[0] for loc in chromosome]
            child_zone_chromosome = deepcopy(zone_chromosome)
            old_soln = zone_chromosome
            new_soln = child_zone_chromosome
            all_gene_options = input_obj.batches
            all_genes_list = list(input_obj.batches.keys())
        else:
            child_chromosome = deepcopy(chromosome)
            old_soln = chromosome
            new_soln = child_chromosome
            all_gene_options = input_obj.genome
            all_genes_list = list(input_obj.genome.keys())

        num_mutations = 1 #!TODO: this was hardcoded to 1 in old MIDAS. Should probably be parameterized.
        chromosome_is_valid = False
        attempts = 0
        while not chromosome_is_valid:
            new_soln = deepcopy(old_soln) #in the case of abortion, start from scratch.
            while new_soln == old_soln:
                for i in range(num_mutations):
                    loc_to_mutate = random.randint(0, len(new_soln)-1) #choose a random gene
                    old_gene = new_soln[loc_to_mutate]
                    gene_options = optools.Constrain_Input.calc_gene_options(all_genes_list, all_gene_options,\
                                                                                LWR_core_parameters, old_soln) #constraint input
                    new_gene = random.choice(gene_options)
                    if new_gene != old_gene:
                        if all_gene_options[new_gene]['map'][loc_to_mutate] == 1:
                            new_soln[loc_to_mutate] = new_gene
            chromosome_is_valid = optools.Constrain_Input.check_constraints(all_genes_list,all_gene_options,\
                                                                            LWR_core_parameters,new_soln)
            if not chromosome_is_valid:
                attempts += 1
                if attempts > 100000:
                    logger.error("Mutate-by-Gene has failed after 100,000 attempts; the Individual will be restored. Consider relaxing the constraints on the input space.")
                    return chromosome

        if input_obj.calculation_type in ["eq_cycle"]:
            #recreate child_chromosome
            child_chromosome = []
            for i in range(len(new_soln)):
                if new_soln[i] == chromosome[i][0]:
                    child_chromosome.append(chromosome[i])
                else:
                    child_chromosome.append((new_soln[i],None))
            child_chromosome = optools.Constrain_Input.EQ_reload_fuel(input_obj.genome,LWR_core_parameters,child_chromosome)

        else: 
            child_chromosome = new_soln

        return child_chromosome


class Cooling_Schedule(object):
    """
    Class for Simulated Annealing cooling schedules.

    THe cooling schedule sets the tolerance for accepting new solutions.
    A high initial temperature indicates accepting new designs even if
    they have a less favorable objective function. Thus the logarithmic cooling
    schedule is favorable for this problem.

    There are two cooling schedules defined below. In both the temperature is
    determined by the current era of a lifetime, represented by a piecewise
    function. The second cooling schedule is identical to the first but with
    two cycles.

    Updated by Jake Mikouchi 04/23/2025
    """

    def __init__(self, generation):
        self.generation = generation

    def exponential_decrease(temperature):
        """
        Logarithmic cooling schedule for simulated annealing. Implementing this because Johnny Klemes cooling schedules seem broken.
        His implementation that I pulled from Github is broken at the least, and there isn't sufficient documentation available to
        understand how to fix it.
        This cooling schedule is about as simple as it can get.
        T = T0*alpha
        Where 0.9 < alpha < 1.0 and 1 < T0 < 10

        This was kept in for legacy reasons 

        Updated by Jake Mikouchi 04/23/2025
        """
        alpha = 0.95 #TODO make this avialble to edit in input file
        if temperature <= 0.0001:
            temperature = 0.0001
        else:
            temperature = temperature * alpha
        return temperature
    
    def linear_update( initial_temperature, current_generation, total_generations):
        """
        linearly updates the temperature
        
        created by Jake Mikouchi 04/23/2025
        """
        temperature = initial_temperature + ((0 - initial_temperature) / total_generations) * (current_generation + 1)
    
        return temperature

    def logarithmic_update(initial_temperature, current_generation):
        """
        Logarithmically updates the temperature
        Note that the user defined inital temperature is used as a contant rather than the actual starting point.
        Literature says this method is rarely used. 
        
        created by Jake Mikouchi 04/23/2025
        """
        temperature = initial_temperature / np.log10(2 + current_generation)
    
        return temperature

    def lam(self, current_generation):
        """
        lam cooling schedule. adaptively updates the temperature acroding to statistics gathered
        during the optimization. This is unique to PSA and is not available for SA.
        Look to https://nstopenresearch.org/articles/2-5 for more information on how it works

        updated by Jake Mikouchi. 4/30/2025
        """

        Buffer_fitnesses = [soln.fitness_value for soln in self.buffer]
        try:
            deviation = statistics.stdev(Buffer_fitnesses)
        except:
            deviation = 1
        
        if deviation == 0: 
            deviation = 0.01

        # calculate move acceptanceratio
        p = self.total_moves / (self.input.num_procs * self.input.population_size)
        if p == 1:
            p = 0.9
        
        Gp = (4 * p * ((1 - p) ** 2)) / ((2 - p) ** 2)

        sk = (1 / self.global_temperature)
        sk1 = sk + (self.input.quality_factor * (1 / deviation) * (1 / ((sk ** 2) * (deviation ** 2)))) * Gp
        temperature = 1 / sk1

        return temperature 

    def lam_set_intial(self):
        """
        calculates the intial temperature using statistics gathered from the buffer
        updated by Jake Mikouchi. 4/30/2025
        """

        Buffer_fitnesses = [soln.fitness_value for soln in self.buffer]
        try:
            deviation = statistics.stdev(Buffer_fitnesses)
        except:
            deviation = 1
        
        if deviation == 0: 
            deviation = 0.01
        
        temperature = deviation * self.input.scaling_factor 

        return temperature

  