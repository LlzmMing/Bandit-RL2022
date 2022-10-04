import numpy as np
from solver import Solver

def softmax(x):
    softmax = np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))
    return softmax

class GradientBandit(Solver):
    def __init__(self, bandit, alpha):
        super(GradientBandit,self).__init__(bandit)
        self.R = np.zeros(bandit.K)
        self.H = np.zeros(bandit.K)
        self.arms = np.arange(bandit.K)
        self.alpha = alpha
    
    def run_one_step(self):
        
        P = softmax(self.H)
        #print(P)
        arm = np.random.choice(self.arms, size=1, p=P)[0]
        r = self.bandit.step(arm)
        #print(arm)
        #print(r)
        #print(self.counts)
        #print(self.R)
        #print(self.H)
        for i in self.arms:
            if (self.counts[i]!=0):
                self.R[i] = self.R[i] + 1/self.counts[i]*(r-self.R[i])
            else:
                self.R[i] = r
            if i == arm:
                self.H[i] = self.H[i] + self.alpha*(r - self.R[i])*(1-P[i])
            else:
                self.H[i] = self.H[i] - self.alpha*(r - self.R[i])*P[i]
        #print(self.R)
        #print(self.H)
        return arm
