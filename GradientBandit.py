import numpy as np
from solver import Solver

def softmax(x):
    softmax = np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))
    return softmax

class GradientBandit(Solver):
    def __init__(self, bandit, alpha,base=False):
        super(GradientBandit,self).__init__(bandit)
        self.R = 0.0
        self.H = np.zeros(bandit.K)
        self.arms = np.arange(bandit.K)
        self.alpha = alpha
        self.steps = 0
        self.base = base
    
    def run_one_step(self):
        
        P = softmax(self.H)

        arm = np.random.choice(self.arms, size=1, p=P)[0]
        r = self.bandit.step(arm)
        
        self.steps += 1
        if self.base:
            self.R = self.R + 1/self.steps*(r-self.R)
        

        for i in self.arms:
            if i == arm:
                self.H[i] = self.H[i] + self.alpha*(r - self.R)*(1-P[i])
            else:
                self.H[i] = self.H[i] - self.alpha*(r - self.R)*P[i]
        #print(self.R)
        #print(self.H)
        return arm
