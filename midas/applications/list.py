import random
from midas.utils.solution_types import Solution

class list_solution(Solution):
    def _init_(self):
        Solution.__init__(self)
        self.type = None
        self.model = None