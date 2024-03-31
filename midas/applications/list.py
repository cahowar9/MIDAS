import random
from multiprocessing import Pool
from midas.utils.solution_types import Solution

class list_solution(Solution):
    """
    Class for creating a solution type for a basic list

    Created by Cole Howard, 2/28/2024
    """
    def __init__(self):
        Solution.__init__(self)
        self.type = None
    
    
