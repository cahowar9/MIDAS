import os
import sys
import copy
import numpy
import h5py
from scipy.stats import spearmanr

"""
This File contains all Termination Criteria Available in MIDAS
"""

class GA_Termination_Criteria(object):


    def __init__(self, file_settings):
        self.Current_Best_Fitness = 0
        self.Previous_Best_Fitness = 0
        self.Consecutive_Generations = 0
        self.Termination_Generations = file_settings['optimization']['Termination_Generations'] - 1
        self.Current_Burnup = [0,0,0,0,0,0]
        self.Previous_Burnup = [0,0,0,0,0,0]
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
            # """

            self.Previous_Best_Fitness = copy.deepcopy(self.Current_Best_Fitness)
            self.Previous_Burnup = copy.deepcopy(self.Current_Burnup)
            self.Current_Burnup = [0,0,0,0,0,0]

            for solution in population:
                with open(f"{solution.name}/{solution.name}_sim.inp","r") as f:
                    # this section retrieves the core from the input file
                    LP = []
                    lines = f.readlines()
                    FAsearch = "FUE.TYP"
                    FAstart = ', '
                    FAend = '/'
                    for line in lines:
                        if FAsearch in line:
                            LineData = (line.split(FAstart)[1].split(FAend)[0])
                            LP.append(LineData.split(" "))

                    CorePatternName = LP[0][0]+LP[1][0]+LP[1][1]+LP[2][0]+LP[2][1]+LP[2][2]+LP[3][0]+LP[3][1]    
                    
                    # retrieves the burnup from hdf5 files
                    f = h5py.File("../../midas/NuScaleModel/Solutions_"+LP[1][0]+".hdf5", 'r')
                    LP_Burnup = f[CorePatternName]["BU"]
                    # for i in range(len(LP_Burnup)):
                    #     self.Current_Burnup[i] += LP_Burnup[i]
                    for i in range(len(CorePatternName)):
                        if int(CorePatternName[i])-2 == 0:
                            self.Current_Burnup[int(CorePatternName[i])-2] += LP_Burnup[i]
                        elif int(CorePatternName[i])-2 == 4 or int(CorePatternName[i])-2 == 7:
                            self.Current_Burnup[int(CorePatternName[i])-2] += LP_Burnup[i]*8
                        else:
                            self.Current_Burnup[int(CorePatternName[i])-2] += LP_Burnup[i]*4

            self.Current_Best_Fitness = spearmanr(self.Previous_Burnup,self.Current_Burnup).statistic
            
            if self.Current_Best_Fitness == self.Previous_Best_Fitness:
                self.Consecutive_Generations += 1
            else:
                self.Consecutive_Generations = 0
            
            if self.Consecutive_Generations < self.Termination_Generations:
                self.Terminate = False
            else:
                self.Terminate = True






        
