import numpy as np
from solver import Solver

class RandomBandit(Solver):
    def __init__(self, bandit):
        super(RandomBandit,self).__init__(bandit)
    
    def run_one_step(self):

        arm = np.random.randint(0,self.bandit.K)
        return arm
