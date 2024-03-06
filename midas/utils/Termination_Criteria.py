import os
import sys
import copy
import numpy

"""
This File contains all Termination Criteria Available in MIDAS
"""

class GA_Termination_Criteria(object):


    def __init__(self):
        self.Current_Best_Fitness = 0
        self.Previous_Best_Fitness = 0
        self.Consecutive_Generations = 0
        pass

    def Consectutive_Fitness(self,population):
        """
        This Termination Criteria is meant to be very simple
        If the Fitness does not increase over a set number of generations, the optimization will automatically stop
        
        Jake Mikouchi 2/28/24
        """

        self.Previous_Best_Fitness = copy.deepcopy(self.Current_Best_Fitness)

        for solution in population:
            if solution.fitness > self.Current_Best_Fitness:
                best_fitness = solution.fitness
                self.Current_Best_Fitness = copy.deepcopy(best_fitness)
        
        if self.Current_Best_Fitness == self.Previous_Best_Fitness:
            self.Consecutive_Generations += 1
        else:
            self.Consecutive_Generations = 0
        
    def Spearman_Fitness(self,population):
        """
        This Termination Criteria utilizes the Spearman rank coefficient
        If the spearman rank does not change over a set number of generations, the optimization will automatically stop
        
        Jake Mikouchi 3/6/24
        """

        self.Previous_Best_Fitness = copy.deepcopy(self.Current_Best_Fitness)

        for solution in population:
            if solution.fitness > self.Current_Best_Fitness:
                best_fitness = solution.fitness
                self.Current_Best_Fitness = copy.deepcopy(best_fitness)
        
        if self.Current_Best_Fitness == self.Previous_Best_Fitness:
            self.Consecutive_Generations += 1
        else:
            self.Consecutive_Generations = 0
