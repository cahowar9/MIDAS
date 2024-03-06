import random
from multiprocessing import Pool
from midas.utils.solution_types import Solution

class list_solution(Solution):
    def __init__(self):
        Solution.__init__(self)
        self.type = None
    
    def add_additional_information(self, settings):
        return(1)
    
    def evaluate(self):
        return(1)
    
