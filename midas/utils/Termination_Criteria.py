import os
import sys
import copy
import numpy

"""
This File contains all Termination Criteria Available in MIDAS
"""

class GA_Termination_Criteria(object):


    def __init__(self, file_settings):
        self.Current_Best_Fitness = 0
        self.Previous_Best_Fitness = 0
        self.Consecutive_Generations = 0
        self.Termination_Generations = file_settings['optimization']['Termination_Generations'] - 1
        self.Terminate = False
        pass

    def Termination_Method(self,population, file_settings):

        if file_settings['optimization']['Termination_Criteria'] == "Consecutive":
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
            
            if self.Consecutive_Generations < self.Termination_Generations:
                self.Terminate = False
            else:
                self.Terminate = True
            
        
        
        if file_settings['optimization']['Termination_Criteria'] == "Spearman":
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
            
            if self.Consecutive_Generations < self.Termination_Generations:
                self.Terminate = False
            else:
                self.Terminate = True
