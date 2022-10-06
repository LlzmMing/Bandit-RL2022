from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from LinUCB import LinUCBPolicy
from Random_baseline import RandomBaseline

def ctr_simulator(K_arms, data_path, d=100, alpha=0.5, is_random=False):
    linucb_policy_object = LinUCBPolicy(K_arms = K_arms, d = d, alpha = alpha)
    random_baseline = RandomBaseline(arms = K_arms)
    time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    
    with open(data_path, "r") as f:
        for line_data in f:
            data_arm = int(line_data.split()[0])
            data_reward = float(line_data.split()[1])
            covariate_string_list = line_data.split()[2:]
            data_x_array = np.array([float(covariate_elem) for covariate_elem in covariate_string_list])
            if not is_random:
                arm_index = linucb_policy_object.select_arm(data_x_array)
                if arm_index + 1 == data_arm:
                    linucb_policy_object.linucb_arms[arm_index].reward_update(data_reward, data_x_array)

                    time_steps += 1
                    cumulative_rewards += data_reward
                    aligned_ctr.append(cumulative_rewards/time_steps)
            
            else:
                random_index = random_baseline.select_arm()
                if random_index + 1 == data_arm:
                    time_steps += 1
                    cumulative_rewards += data_reward
                    aligned_ctr.append(cumulative_rewards/time_steps)
                    
    return aligned_ctr

alpha_input1 = 0.5
alpha_input2 = 1.0
alpha_input3 = 1.5
data_path = "linucb_disjoint_dataset.txt"
aligned_ctr1 = ctr_simulator(K_arms = 10, d = 100, alpha = alpha_input1, data_path = data_path)
aligned_ctr2 = ctr_simulator(K_arms = 10, d = 100, alpha = alpha_input2, data_path = data_path)
aligned_ctr3 = ctr_simulator(K_arms = 10, d = 100, alpha = alpha_input3, data_path = data_path)
aligned_ctr_base = ctr_simulator(K_arms = 10, data_path = data_path, is_random = True)
plt.plot(aligned_ctr1,label="alpha=0.5")
plt.plot(aligned_ctr2,label="alpha=1.0")
plt.plot(aligned_ctr3,label="alpha=1.5")
plt.plot(aligned_ctr_base, label = "random")
plt.title("LinUCB")
plt.legend()
plt.show()