import numpy as np
import matplotlib.pyplot as plt
from MAB import BernoulliBandit
from GradientBandit import GradientBandit
from RandomBandit import RandomBandit
from SoftmaxBandit import SoftmaxBandit
from TPS import ThompsonSampling
from UCB_v1 import UCB

np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个 %d臂伯努利老虎机" % K)
print("获得奖励概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

print('the true q-values for each arm are:')
for value in bandit_10_arm.probs:
    print('{}'.format(np.format_float_positional(value,precision=4)),end = ' ')
print()

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

def test_random_solver():
    np.random.seed(1)
    random_solver = RandomBandit(bandit_10_arm)
    random_solver.run(5000)
    print('RandomSolver的累积懊悔为：', random_solver.regret)
    plot_results([random_solver],["RandomSolver"])

def test_softmax_solver():
    np.random.seed(1)
    softmax_solver = SoftmaxBandit(bandit_10_arm)
    softmax_solver.run(5000)
    print("SoftmaxSolver的累积懊悔为:", softmax_solver.regret)
    plot_results([softmax_solver],["SoftmaxSolver"])

def test_gradient_solver():
    np.random.seed(1)
    gradient_solver_1_base = GradientBandit(bandit_10_arm, alpha = 0.1, base = True)
    gradient_solver_1_nobase = GradientBandit(bandit_10_arm, alpha = 0.1)
    gradient_solver_4_base = GradientBandit(bandit_10_arm, alpha = 0.4, base = True)
    gradient_solver_4_nobase = GradientBandit(bandit_10_arm, alpha = 0.4)
    gradient_solver_1_base.run(5000)
    print("alpha=0.1,有baseline的GradientSolver累积懊悔为：",gradient_solver_1_base.regret)
    gradient_solver_1_nobase.run(5000)
    print("alpha=0.1,无baseline的GradientSolver累积懊悔为：",gradient_solver_1_nobase.regret)
    gradient_solver_4_base.run(5000)
    print("alpha=0.4,有baseline的GradientSolver累积懊悔为：",gradient_solver_4_base.regret)
    gradient_solver_4_nobase.run(5000)
    print("alpha=0.4,无baseline的GradientSolver累积懊悔为：",gradient_solver_4_nobase.regret)
    solvers = [gradient_solver_1_base,gradient_solver_1_nobase,gradient_solver_4_base,gradient_solver_4_nobase]
    names = ["GradientBandit_0.1_baseline","GradientBandit_0.1_nobaseline","GradientBandit_0.4_baseline","GradientBandit_0.4_nobaseline"]
    plot_results(solvers, names)

def test_tompson_sampling_solver():
    np.random.seed(1)
    thompsom_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompsom_sampling_solver.run(5000)
    print('TPS的累积懊悔为：', thompsom_sampling_solver.regret)
    plot_results([thompsom_sampling_solver], ['TPS'])

def test_UCB_solver():
    np.random.seed(1)
    UCB_solver_1 = UCB(bandit_10_arm, coef= 1)
    UCB_solver_2 = UCB(bandit_10_arm, coef = 0.5)
    UCB_solver_3 = UCB(bandit_10_arm, coef = 0.1)
    UCB_solver_4 = UCB(bandit_10_arm, coef = 0)
    solvers = [UCB_solver_1,UCB_solver_2,UCB_solver_3,UCB_solver_4]
    for solver in solvers:
        solver.run(8000)
    plot_results(solvers, ["UCB-coef=1","UCB-coef=0.5","UCB-coef=0.1", "greedy UCB-coef=0"])

if __name__ == "__main__":
    test_random_solver()
    test_softmax_solver()
    test_gradient_solver()
    test_tompson_sampling_solver()
    test_UCB_solver()


