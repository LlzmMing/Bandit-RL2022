import numpy as np

class LinUCBDisJointArm():
    def __init__(self, arm_index, d, alpha):
        self.arm_index = arm_index
        self.alpha = alpha
        self.A = np.identity(d)
        self.b = np.zeros([d,1])
    
    def calc_UCB(self, x_array):
        A_inv = np.linalg.inv(self.A)
        self.theta = np.dot(A_inv, self.b)
        x = x_array.reshape([-1,1])
        p = np.dot(self.theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv,x)))
        
        return p
    
    def reward_update(self, reward, x_array):
        x = x_array.reshape([-1,1])
        self.A += np.dot(x, x.T)
        self.b += reward * x        

class LinUCBPolicy():
    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [LinUCBDisJointArm(arm_index = i, d = d, alpha = alpha) for i in range(K_arms)]
    
    def select_arm(self, x_array):
        highest_ucb = -1
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)
            if arm_ucb > highest_ucb:
                highest_ucb = arm_ucb
                candidate_arms = [arm_index]

            if arm_ucb == highest_ucb:
                
                candidate_arms.append(arm_index)
        chosen_arm = np.random.choice(candidate_arms)
        
        return chosen_arm