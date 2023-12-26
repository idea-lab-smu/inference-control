import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

import learning_rate as lr
from state_history import StateManager


initial_var = 0.0
threshold = 0.005
initial_max_var = threshold


class chainEnv():

	def __init__(self, size):
		
		self.size = size
		self.actions = size - 1

		#self.state = []
		self.state = [initial_var for i in range(self.size)]
		self.new_state = [initial_var for i in range(self.size)]
		self.max_state = [[0, 0] for i in range(self.size)]
		self.hit_rate = [0 for i in range(self.size)]

		self.reward = 0
		self.done = False
		self.state_manager = StateManager(size)
		#self.state = self.reset([])
		
	def reset(self, init_condition = []):
	
		var_uni = [1 / 2.0 for i in range(self.size)]
		_, init_state = self.state_manager.compute_dirichlet_stats(var_uni)
		init_history = [0 for i in range(self.size)]
		self.state = init_state #+ init_history

		self.new_state = [initial_var for i in range(self.size)]
		self.max_state = [[0, 0] for i in range(self.size)]
		self.hit_rate = [0 for i in range(self.size)]

		self.reward = 0
		self.done = False
		self.prev_state = 0
		self.state_manager.reset()

		return self.state


	def set_initial_history(self, alpha):

		self.state_manager.set_initial_history(alpha)


	def is_decreasing(self, prev, next):

		neg = (next - prev) < 0
		cross = (next <= threshold)
	
		res = neg and cross
	
		return res

	
	# for conjugacy and update rule
	def is_terminal(self, variances):

		valid = 0
		ret = False
		reward = -1

		for v in variances:
			if v <= threshold:
				valid += 1

		history = self.state_manager.get_visit_history()
		nonnovel = history[:self.size - 1]
		nonnovel = history[:]

		if valid == self.size and min(nonnovel) >= 1:
			ret = True
			reward = 1
		
		return ret, reward
	
	
	def interpret(self, action):
	
		state = self.state
		sign = 1
		steps = 0
		
		if action < state:
			sign = -1
			steps = state - action
		elif action == state:
			sign = 1
			steps = 1 
		else:
			sign = 1
			steps = action - state + 1
			
		return sign * steps


	def compute_uncertainty(self, action):

		learning_rate, variances = self.state_manager.update(action)

		learning_rate, variances = lr.compute_learning_rate(distribution)

		return learning_rate, variances


	def merge(self, dest, source, sp, length):

		ret = []

		for sp in range(length):

			dest[sp] = source[sp]			

		ret = dest

		return ret


	def step(self, action):
		
		visits = self.state_manager.get_visit_history()
		learning_rate, variances = self.state_manager.update(action)		
		new_visits = self.state_manager.get_visit_history()

		self.state = self.new_state
		self.new_state = variances# + new_visits
		self.done, self.reward = self.is_terminal(variances)
		
		return self.new_state, self.reward, self.done, learning_rate

