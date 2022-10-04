import numpy as np
import matplotlib.pyplot as plt
from MAB import BernoulliBandit
from GradientBandit import GradientBandit
from TPS import ThompsonSampling

np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个 %d臂伯努利老虎机" % K)
print("获得奖励概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

np.random.seed(1)
gradient_solver = GradientBandit(bandit_10_arm, alpha = 0.1)


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表，列表中每个元素是一种特定的策略。而solver_names也是一个列表，包含每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    gradient_solver.run(5000)
    plot_results([gradient_solver], ["GradientBandit"])

    np.random.seed(1)
    thompsom_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompsom_sampling_solver.run(5000)
    print('TPS的累积懊悔为：', thompsom_sampling_solver.regret)
    plot_results([thompsom_sampling_solver], ['TPS'])


