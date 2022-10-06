'''
Chen Menglong
October 3, 2022
File that implement the Upper Confidence Bound Algorithm
version 1
'''

import numpy as np
from solver import Solver

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0, bias = 1):
        super().__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * bandit.K)
        self.coef = coef
        self.bias = bias #a small bias used to avoid the case when N_t(a) is 0
    
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count)/(self.counts + self.bias)) #Q_t(a) + c*sqrt(ln(t)/(N_t(a)+bias))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1/(self.counts[k] + 1) * (r - self.estimates[k]) #update Q_t
        return k

        return 