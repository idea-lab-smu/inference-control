import numpy as np
import math

class StateManager(object):

	def __init__(self, space_size):
	
		# initial probability 
		# - on the uniform distribution in this case
		self.init_prob = 2.0
		self.env_size = space_size

		# visit history buffer
		self.visits = [self.init_prob for i in range(self.env_size)]
		self.visit_history = [0 for i in range(self.env_size)]		
		self.action_history = []

		# the index of the novel cue in this setting
		self.cue = 2
		self.learning_rate = self.compute_learning_rate([1 for i in range(self.env_size)])


	def reset(self):

		# visit history buffer
		self.visits = [self.init_prob for i in range(self.env_size)]
		self.visit_history = [0 for i in range(self.env_size)]		
		self.action_history = []
		self.learning_rate = self.compute_learning_rate([1 for i in range(self.env_size)])


	def compute_dirichlet_stats(self, alpha):

		alpha_sum = sum(alpha)
		alpha_mean = []
		alpha_var = []

		for i in range(len(alpha)):
			mean = alpha[i] / alpha_sum
			alpha_mean.insert(i, mean)

			variance = (alpha[i] * (alpha_sum - alpha[i])) / (pow(alpha_sum, 2) * (alpha_sum + 1.0))
			alpha_var.insert(i, variance)

		return alpha_mean, alpha_var		
		

	def compute_learning_rate(self, alpha_var):

		tau = 100 # amplify 100 times
		lrate = []
		avsum = 0
		i = 0

		for av in alpha_var:
			avsum = avsum + math.exp(tau * av)

		for av in alpha_var:
			lr = math.exp(tau * av) / float(avsum)
			lrate.append(lr)

		return lrate


	def get_visit_history(self):

		return self.visit_history


	def get_state(self):

		return self.visits

	'''

	def utmostise_learning_rate_ext(self, lr, action):

		novel = 2
		ret_lr = lr[:]
		temp_lr = lr[:]

		if action == novel:

			temp = temp_lr.pop(action)
			max_lr = max(temp_lr)

			for i in range(len(lr)):
				if i < novel:
					ret_lr[i] = max_lr - lr[i]
				else:
					ret_lr[i] = max_lr + lr[i]
			
		else:

			max_lr = temp_lr[novel]

			for i in range(len(lr)):
				if i != novel:
					if i == action:
						ret_lr[i] = max_lr + lr[i]
					else:
						ret_lr[i] = 0
				else:
					ret_lr[i] = max_lr - lr[i]

		return ret_lr
	'''


	def utmostise_learning_rate(self, lr, action):

		novel = 4
		pair = [[0, 2], [1, 3]]

		ret_lr = lr[action]
		temp_lr = lr[:]

		if action == novel:
			temp = temp_lr.pop(action)
		else:
			idx = action % 2	
			for val in pair[idx]:
				temp_lr.pop(val)

		max_lr = max(temp_lr)
		ret_lr += max_lr
		return ret_lr

	
	def update(self, action):

		novel = 2

		self.action_history.append(action)
		self.visit_history[action] += 1

		cnt = len(self.action_history)

		'''
		for idx in range(self.env_size):

			if action == idx:
				sign = 1
			else:
				sign = -1

			delta = sign * self.learning_rate[idx]
			self.visits[idx] = max(0.0, self.visits[idx] + delta)
		'''
		'''
		if action == novel:
			sign = 1
		else:
			sign = -1
		
		if cnt > 1:

			delta = sign * self.learning_rate[action]
			self.visits[action] = max(0.0, self.visits[action] + delta)
		'''

		#cur_lr = self.utmostise_learning_rate(self.learning_rate, action)
		#self.visits[action] += (cur_lr * 2)
		# no oneshot effect
		self.visits[action] += 1
		
		dc_mean, dc_var = self.compute_dirichlet_stats(self.visits)

		#if (cnt > 0) and ((cnt % 5) == 0):
		self.learning_rate = self.compute_learning_rate(dc_var)

		return self.learning_rate, dc_var

		
