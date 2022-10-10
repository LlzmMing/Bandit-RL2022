'''
CML addapted from Stanford cs234 assignment 4 winter 2022.
Oct 10, 2022
'''
from abc import ABC, abstractmethod

import numpy as np
import csv
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import load_data, LABEL_KEY

import pdb

def dose_class(weekly_dose):
	if weekly_dose < 21:
		return 'low'
	elif 21 <= weekly_dose and weekly_dose <= 49:
		return 'medium'
	else:
		return 'high'


# Base classes
class BanditPolicy(ABC):
	@abstractmethod
	def choose(self, x): pass

	@abstractmethod
	def update(self, x, a, r): pass

class StaticPolicy(BanditPolicy):
	def update(self, x, a, r): pass

class RandomPolicy(StaticPolicy):
	def __init__(self, probs=None):
		self.probs = probs if probs is not None else [1./3., 1./3., 1./3.]

	def choose(self, x):
		return np.random.choice(('low', 'medium', 'high'), p=self.probs)

# Baselines
class FixedDosePolicy(StaticPolicy):
	def choose(self, x):
		"""
		Args:
			x: Dictionary containing the possible patient features. 
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the fixed dose algorithm.
		"""
		#######################################################
		#########   YOUR CODE HERE - ~1 lines.   #############
		return 'medium'
		#######################################################
		######### 

class ClinicalDosingPolicy(StaticPolicy):
	def choose(self, x):
		"""
		Args:
			x: Dictionary containing the possible patient features. 
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the Clinical Dosing algorithm.

		Hint:
			- You may need to do a little data processing here. 
			- Look at the "main" function to see the key values of the features you can use. The
				age in decades is implemented for you as an example.
			- You can treat Unknown race as missing or mixed race.
			- Use dose_class() implemented for you. 
		"""
		age_in_decades = x['Age in decades']

		#######################################################
		#########   YOUR CODE HERE - ~2-10 lines.   #############
		sqrtWeeklyDose = 4.0376 - 0.2546*age_in_decades + 0.0118 * x['Height (cm)'] + 0.0134 * x['Weight (kg)'] - 0.6752 * x['Asian'] + 0.4060 * x['Black'] + 0.0443 * x['Unknown race'] + \
			1.2799 * (x['Carbamazepine (Tegretol)'] or x['Phenytoin (Dilantin)'] or x['Rifampin or Rifampicin']) - 0.5695 * x['Amiodarone (Cordarone)']
		return dose_class(sqrtWeeklyDose**2)
		#######################################################
		######### 

# Upper Confidence Bound Linear Bandit
class LinUCB(BanditPolicy):
	def __init__(self, n_arms, features, alpha=1.):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation" 

		Args:
			n_arms: int, the number of different arms/ actions the algorithm can take 
			features: list of strings, contains the patient features to use 
			alpha: float, hyperparameter for step size. 
		
		TODO:
		Please initialize the following internal variables for the Disjoint Linear Upper Confidence Bound Bandit algorithm. 
		Please refer to the paper to understadard what they are. 
		Please feel free to add additional internal variables if you need them, but they are not necessary. 

		Hints:
		Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
		"""
		#######################################################
		#########   YOUR CODE HERE - ~5 lines.   #############
		self.n_arms = n_arms
		self.features = features
		self.n_features = len(features)
		self.arms = ['low', 'medium', 'high']
		self.alpha = alpha
		self.A = {arm: np.identity(self.n_features) for arm in self.arms}
		self.b = {arm: np.zeros((self.n_features,1)) for arm in self.arms}
		# self.A = [np.eye(len(self.features))]*n_arms
		# self.b = [np.zeros((self.n_features,1))]*n_arms
		### improve efficiency by reusing A_inv as introduced in the paper -- CML
		self.A_invs = {arm: np.linalg.inv(self.A[arm]) for arm in self.arms}
		#######################################################
		#########          END YOUR CODE.          ############

	def choose(self, x):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation"

		Args:
			x: Dictionary containing the possible patient features. 
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm. 
		"""
		#######################################################
		#########   YOUR CODE HERE - ~7 lines.   #############
		featVec = np.array([x[feat] for feat in self.features]).reshape((self.n_features,1))
		thetas = {arm: np.matmul(self.A_invs[arm], self.b[arm]) for arm in self.arms}
		assert thetas['low'].shape == (self.n_features,1)
		scores = {arm: np.matmul(thetas[arm].T, featVec)+self.alpha*np.sqrt(np.matmul(np.matmul(featVec.T, self.A_invs[arm]), featVec)) for arm in self.arms}
		return max(scores.items(), key=lambda x:x[1])[0]
		#######################################################
		######### 

	def update(self, x, a, r):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation"
			
		Args:
			x: Dictionary containing the possible patient features. 
			a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
			r: the reward you recieved for that action
		Returns:
			Nothing

		TODO:
		Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm. 

		Hint: Which parameters should you update?
		"""
		#######################################################
		#########   YOUR CODE HERE - ~4 lines.   #############
		featVec = np.array([x[feat] for feat in self.features]).reshape((self.n_features,1))
		# arm_num = ['low', 'medium', 'high'].index(a)
		self.A[a] += featVec @ featVec.T ###Sadly, errors like feat.T @ feat cannot be found easily.
		self.b[a] += r * featVec
		assert self.b[a].shape == (self.n_features,1)
		self.A_invs[a] = np.linalg.inv(self.A[a])
		#######################################################
		#########          END YOUR CODE.          ############

# eGreedy Linear bandit
class eGreedyLinB(LinUCB):
	def __init__(self, n_arms, features, alpha=1.):
		super(eGreedyLinB, self).__init__(n_arms, features, alpha=1.)
		self.time = 0  
	def choose(self, x):
		"""
		Args:
			x: Dictionary containing the possible patient features. 
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Instead of using the Upper Confidence Bound to find which action to take, 
		compute the probability of each action using a simple dot product between Theta & the input features.
		Then use an epsilion greedy algorithm to choose the action. 
		Use the value of epsilon provided
		"""
		
		self.time += 1 
		epsilon = float(1./self.time)* self.alpha
		#######################################################
		#########   YOUR CODE HERE - ~7 lines.   #############
		featVec = np.array([x[feat] for feat in self.features]).reshape((self.n_features,1))
		thetas = {arm: np.matmul(self.A_invs[arm], self.b[arm]) for arm in self.arms}
		assert thetas['low'].shape == (self.n_features,1)
		scores = {arm: np.matmul(thetas[arm].T, featVec) for arm in self.arms}
		random_value = np.random.uniform()
		rand_policy = RandomPolicy()
		if random_value < 1 - epsilon:
			action = max(scores.items(), key=lambda x:x[1])[0]
		else:
			action = rand_policy.choose(x)
		return action
		#######################################################
		######### 


# Thompson Sampling
class ThomSampB(BanditPolicy):
	def __init__(self, n_arms, features, alpha=1.):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs" 

		Args:
			n_arms: int, the number of different arms/ actions the algorithm can take 
			features: list of strings, contains the patient features to use 
			alpha: float, hyperparameter for step size.
		
		TODO:
		Please initialize the following internal variables for the Disjoint Thompson Sampling Bandit algorithm. 
		Please refer to the paper to understadard what they are. 
		Please feel free to add additional internal variables if you need them, but they are not necessary. 

		Hints:
			- Keep track of a seperate B, mu, f for each action (this is what the Disjoint in the algorithm name means)
			- Unlike in section 2.2 in the paper where they sample a single mu_tilde, we'll sample a mu_tilde for each arm 
				based on our saved B, f, and mu values for each arm. Also, when we update, we only update the B, f, and mu
				values for the arm that we selected
			- What the paper refers to as b in our case is the medical features vector
			- The paper uses a summation (from time =0, .., t-1) to compute the model paramters at time step (t),
				however if you can't access prior data how might one store the result from the prior time steps.
		
		"""

		#######################################################
		#########   YOUR CODE HERE - ~6 lines.   #############
		self.n_arms = n_arms
		self.features = features
		#Simply use aplha for the v mentioned in the paper
		self.v2 = alpha 
		self.arms = ['low', 'medium', 'high']
		self.n_features = len(features)
		self.B = {arm: np.eye(self.n_features) for arm in self.arms}

		#Variable used to keep track of data needed to compute mu
		self.f = {arm: np.zeros((self.n_features,)) for arm in self.arms}

		#You can actually compute mu from B and f at each time step. So you don't have to use this.
		self.mu = {arm: np.zeros((self.n_features,)) for arm in self.arms}  ### I will not use this.
		### I can still cache B_inv like linUCB to save time--CML
		self.B_invs = {arm: np.eye(self.n_features) for arm in self.arms}
		#######################################################
		#########          END YOUR CODE.          ############



	def choose(self, x):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs" 

		Args:
			x: Dictionary containing the possible patient features. 
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the "forward pass" for Disjoint Thompson Sampling Bandit algorithm. 
		Please use the gaussian distribution like they do in the paper
		"""

		#######################################################
		#########   YOUR CODE HERE - ~8 lines.   #############
		featVec = np.array([x[feat] for feat in self.features]).reshape((self.n_features,1))
		self.mu = {arm: self.B_invs[arm] @ self.f[arm] for arm in self.arms}
		# multivariate_normal needs input of 1D array for the mean.
		self.mu_sample = {arm: np.random.multivariate_normal(self.mu[arm],self.v2 * self.B_invs[arm]) for arm in self.arms}
		arm_scores = {arm: featVec.T @ self.mu_sample[arm] for arm in self.arms}
		action = max(arm_scores.items(), key = lambda x:x[1])[0]
		return action
		#######################################################
		#########          END YOUR CODE.          ############

	def update(self, x, a, r):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs" 
			
		Args:
			x: Dictionary containing the possible patient features. 
			a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
			r: the reward you recieved for that action
		Returns:
			Nothing

		TODO:
		Please implement the update step for Disjoint Thompson Sampling Bandit algorithm. 
		Please use the gaussian distribution like they do in the paper

		Hint: Which parameters should you update?
		"""

		#######################################################
		#########   YOUR CODE HERE - ~6 lines.   #############
		featVec = np.array([x[feat] for feat in self.features]).reshape((self.n_features,1))
		self.B[a] += featVec @ featVec.T
		self.f[a] += r * featVec.reshape((-1,))
		self.B_invs[a] = np.linalg.inv(self.B[a])
		#######################################################
		#########          END YOUR CODE.          ############

###linear optimal
class LinOpt(BanditPolicy):
	def __init__(self, n_arms, features, training=True):
		"""
		essentially linUCB with alpha=0, with training argument to stop the update of parameters in evaluation.
		Args:
			n_arms: int, the number of different arms/ actions the algorithm can take 
			features: list of strings, contains the patient features to use 
			training: Boolean, if the parameters needs to be updated online.
		"""
		self.n_arms = n_arms
		self.features = features
		self.training = training
		self.arms = ['low', 'medium', 'high']
		self.n_features = len(features)
		self.A = {arm: np.identity(self.n_features) for arm in self.arms}
		self.b = {arm: np.zeros((self.n_features,1)) for arm in self.arms}
		self.A_invs = {arm: np.linalg.inv(self.A[arm]) for arm in self.arms}
	def train(self):
		self.training=True
	def evaluate(self):
		self.training=False
	def choose(self, x):
		"""
		Args:
			x: Dictionary containing the possible patient features. 
		Returns:
			output: string containing one of ('low', 'medium', 'high')
		"""
		featVec = np.array([x[feat] for feat in self.features]).reshape((self.n_features,1))
		thetas = {arm: np.matmul(self.A_invs[arm], self.b[arm]) for arm in self.arms}
		assert thetas['low'].shape == (self.n_features,1)
		scores = {arm: np.matmul(thetas[arm].T, featVec) for arm in self.arms}
		return max(scores.items(), key=lambda x:x[1])[0]
	def update(self, x, a, r):
		"""
		Args:
			x: Dictionary containing the possible patient features. 
			a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
			r: the reward you recieved for that action
		Returns:
			Nothing
		"""
		if self.training==False:
			return
		featVec = np.array([x[feat] for feat in self.features]).reshape((self.n_features,1))
		# arm_num = ['low', 'medium', 'high'].index(a)
		self.A[a] += featVec @ featVec.T ###Sadly, errors like feat.T @ feat cannot be found easily.
		self.b[a] += r * featVec
		assert self.b[a].shape == (self.n_features,1)
		self.A_invs[a] = np.linalg.inv(self.A[a])
		#######################################################
		#########          END YOUR CODE.          ############

def fit_linear_optimal(data, learner, large_error_penalty=False):
	# Shuffle
	data = data.sample(frac=1)
	T = len(data)
	n_egregious = 0
	correct = np.zeros(T, dtype=bool)
	for t in range(T):
		x = dict(data.iloc[t])
		label = x.pop(LABEL_KEY)
		for action in learner.arms:###essentially try every arm, and update the parameters of every arm.
			correct[t] = (action == dose_class(label))
			reward = int(correct[t]) - 1
			if (action == 'low' and dose_class(label) == 'high') or (action == 'high' and dose_class(label) == 'low'):
				n_egregious += 1
				reward = large_error_penalty
			learner.update(x, action, reward)
	return

def run(data, learner, large_error_penalty=False):
	# Shuffle
	data = data.sample(frac=1)
	T = len(data)
	n_egregious = 0
	correct = np.zeros(T, dtype=bool)
	for t in range(T):
		x = dict(data.iloc[t])
		label = x.pop(LABEL_KEY)
		action = learner.choose(x)
		correct[t] = (action == dose_class(label))
		reward = int(correct[t]) - 1
		if (action == 'low' and dose_class(label) == 'high') or (action == 'high' and dose_class(label) == 'low'):
			n_egregious += 1
			reward = large_error_penalty
		learner.update(x, action, reward)

	return {
		'total_fraction_correct': np.mean(correct),
		'average_fraction_incorrect': np.mean([
			np.mean(~correct[:t]) for t in range(1,T) ]),
		'fraction_incorrect_per_time': [
			np.mean(~correct[:t]) for t in range(1,T)],
		'fraction_egregious': float(n_egregious) / T
	}

def main(args):
	data = load_data()

	frac_incorrect = []
	features = [
			'Age in decades',
			'Height (cm)', 'Weight (kg)',
			'Male', 'Female',
			'Asian', 'Black', 'White', 'Unknown race',
			'Carbamazepine (Tegretol)',
			'Phenytoin (Dilantin)',
			'Rifampin or Rifampicin',
			'Amiodarone (Cordarone)'
		]

	extra_features = [ 
			'VKORC1AG', 'VKORC1AA', 'VKORC1UN', 
			'CYP2C912', 'CYP2C913', 'CYP2C922', 
			'CYP2C923', 'CYP2C933', 'CYP2C9UN'
		]

	features = features + extra_features

	if args.run_fixed:
		avg = []
		for i in range(args.runs): 
			print('Running fixed')
			results = run(data, FixedDosePolicy())
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("Fixed", np.mean(np.asarray(avg),0)))

	if args.run_clinical:
		avg = []
		for i in range(args.runs): 
			print('Runnining clinical')
			results = run(data, ClinicalDosingPolicy())
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("Clinical", np.mean(np.asarray(avg),0)))

	if args.run_linucb: 
		avg = []
		for i in range(args.runs): 
			print('Running LinUCB bandit')
			results = run(data, LinUCB(3, features, alpha=args.alpha), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("LinUCB", np.mean(np.asarray(avg),0)))

	if args.run_egreedy: 
		avg = []
		for i in range(args.runs): 
			print('Running eGreedy bandit')
			results = run(data, eGreedyLinB(3, features, alpha=args.ep), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("eGreedy", np.mean(np.asarray(avg),0)))

	if args.run_thompson: 
		avg = []
		for i in range(args.runs): 
			print('Running Thompson Sampling bandit')
			results = run(data, ThomSampB(3, features, alpha=args.v2), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("Thompson", np.mean(np.asarray(avg),0)))
	###add random runner
	if args.run_random:###Interestingly, we cannot write args.run-random, which generates AttributeError: 'Namespace' object has no attribute 'run'
		avg = []
		for i in range(args.runs): 
			print('Running Random bandit')
			results = run(data, RandomPolicy(), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("Random", np.mean(np.asarray(avg),0)))
	###
	###add deterministic linear optimal runner
	if args.run_linear_optimal:
		###first, fit an optimal linear learner with ridge regression
		print('Fitting Opt bandit')
		optLearner = LinOpt(3,features)
		optLearner.train()
		fit_linear_optimal(data,optLearner,large_error_penalty=args.large_error_penalty)
		optLearner.evaluate()
		print('Is it still training?',optLearner.training)
	
		avg = []
		for i in range(args.runs): 
			print('Running Opt bandit')
			results = run(data, optLearner, large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("LinOpt", np.mean(np.asarray(avg),0)))
	
	###add lingreedy
	if args.run_lingreedy:
		avg = []
		for i in range(args.runs): 
			print('Running LinGreedy bandit')
			results = run(data, LinUCB(3, features, alpha=0), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("LinGreedy", np.mean(np.asarray(avg),0)))


	os.makedirs('results', exist_ok=True)
	if frac_incorrect != []:
		for algorithm, results in frac_incorrect:
			with open(f'results/{algorithm}.csv', 'w') as f:
				csv.writer(f).writerows(results.reshape(-1, 1).tolist())
	frac_incorrect = []
	for filename in os.listdir('results'):
		if filename.endswith('.csv'):
			algorithm = filename.split('.')[0]
			with open(os.path.join('results', filename), 'r') as f:
				### for some reason, there are empty lists in raw_l, 
				### and without removing them numpy will report errors.--CML
				raw_l = list(csv.reader(f))
				better_l = [li for li in raw_l if li != []]
				frac_incorrect.append((algorithm, np.array(better_l).astype('float64').squeeze()))
				# frac_incorrect.append((algorithm, np.array(list(csv.reader(f))).astype('float64').squeeze()))			
	plt.xlabel("examples seen")
	plt.ylabel("fraction_incorrect")
	legend = []	
	for name, values in frac_incorrect:
		legend.append(name)
		plt.plot(values[10:])
	# plt.ylim(0.0, 1.0)
	plt.ylim(0.25, 0.7) ### to show more clearly.
	plt.legend(legend)
	plt.savefig(os.path.join('results', 'fraction_incorrect-20runs.png'))
	
if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('--run-fixed', action='store_true')
	parser.add_argument('--run-clinical', action='store_true')
	parser.add_argument('--run-linucb', action='store_true')
	parser.add_argument('--run-egreedy', action='store_true')
	parser.add_argument('--run-thompson', action='store_true')
	parser.add_argument('--alpha', type=float, default=1.)
	parser.add_argument('--ep', type=float, default=1)
	parser.add_argument('--v2', type=float, default=0.001)
	parser.add_argument('--runs', type=int, default=20)
	parser.add_argument('--large-error-penalty', type=float, default=-1)
	###add random
	parser.add_argument('--run-random',action='store_true')
	###add performance upper bound by fitting an optimal linear classifier and then run on the data again.
	parser.add_argument('--run-linear-optimal',action='store_true')
	###add lingreedy
	parser.add_argument('--run-lingreedy',action='store_true')
	args = parser.parse_args()
	main(args)