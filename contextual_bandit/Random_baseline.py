import numpy as np

class RandomBaseline():
    def __init__(self, arms):
        self.arms = arms
    
    def select_arm(self):
        return np.random.randint(0,self.arms)