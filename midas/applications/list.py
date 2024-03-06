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
    
    def add_additional_information(self, settings):
        """
        Adds information into the list solution type. For now, only needs the length of each list in the population

        Created by Cole Howard, 2/28/2024
        """
        info = settings['genome']['list_data']

        if 'list_length' in info:
            self.list_length = info['list_length']
        
    
    def evaluate(self):
        return(1)
    
