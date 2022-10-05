import numpy as np
from solver import Solver

def softmax(x):
    softmax = np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))
    return softmax

class SoftmaxBandit(Solver):
    def __init__(self, bandit):
        super(SoftmaxBandit,self).__init__(bandit)
        self.arms = np.arange(bandit.K)
        self.R = np.zeros(bandit.K)
        self.step = np.zeros(bandit.K)

    def run_one_step(self):
        
        P = softmax(self.R)
        arm = np.random.choice(self.arms, size=1, p=P)[0]
        r = self.bandit.step(arm)
        
        self.step[arm] += 1
        self.R[arm] = self.R[arm] + 1/self.step[arm]*(r-self.R[arm])
        
        return arm
