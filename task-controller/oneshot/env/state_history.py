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
		self.novel = 4
		self.learn_from_absence = 0

		_, pv = self.compute_dirichlet_stats(self.visits)
		self.learning_rate = self.compute_learning_rate(pv)


	def reset(self):

		# visit history buffer
		self.visits = [self.init_prob for i in range(self.env_size)]
		self.visit_history = [0 for i in range(self.env_size)]		
		self.action_history = []

		_, pv = self.compute_dirichlet_stats(self.visits)
		self.learning_rate = self.compute_learning_rate(pv)


	def compute_dirichlet_stats(self, alpha):

		#alpha_sum = sum(alpha)
		alpha_sum = max(sum(alpha), 1e-20)
		alpha_mean = []
		alpha_var = []

		for i in range(len(alpha)):
			mean = alpha[i] / alpha_sum
			alpha_mean.insert(i, mean)

			variance = (alpha[i] * (alpha_sum - alpha[i])) / float((pow(alpha_sum, 2) * (alpha_sum + 1.0)))
			alpha_var.insert(i, variance)

		return alpha_mean, alpha_var		
		

	def compute_learning_rate(self, alpha_var):

		tau = 126.20 # amplify 100 times
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


	def get_reward(self, action, novel):

		ret = 1.5

		if action == novel:
			ret = 7.5
		else:
			if action % 2 == 0:
				ret = 1.5
			else:
				ret = 2.5

		return ret


	def compute_saliency(self, lr, action):

		
		trial = [0 for i in range(self.env_size)]
		saliency = [0 for i in range(self.env_size)]		

		trial[action] += 1.5
		cur_lr = self.learning_rate

		for k in range(self.env_size):

			sign = 1

			rwd_val = self.get_reward(action, self.novel)

			cue_cnt = min(trial[k], 1000)

			if cue_cnt == 0:
				cue_cnt = 1 

				#if action == self.novel and k != action:
				#	sign = -1

				if self.learn_from_absence == 1:	# strengthened the oneshot effect!
					rwd_val = sign * rwd_val * (-0.5)
				else:
					rwd_val = sign * rwd_val * (-1e-5)

			rwd_val = rwd_val * 0.5
			rwd_val_up = (-1) * 1 * rwd_val 
			saliency[k] = cue_cnt * cur_lr[k] * rwd_val_up

		return saliency

	
	def update(self, action):

		#if sum(self.visit_history) == 0 and self.novel == -1:
		#	self.novel = action

		self.action_history.append(action)
		self.visit_history[action] += 1

		saliency = self.compute_saliency(self.learning_rate, action)
		for i in range(self.env_size):
			saliency[i] = max((-1) * self.visits[i], saliency[i])
			self.visits[i] += (saliency[i])		

		dc_mean, dc_var = self.compute_dirichlet_stats(self.visits)
		self.learning_rate = self.compute_learning_rate(dc_var)

		'''
		print '\tsaliency', saliency
		print '\talpha', self.visits 
		print '\tvariance', dc_var
		print '\tgamma', self.learning_rate,'\n'
		'''

		return self.learning_rate, dc_var

		
