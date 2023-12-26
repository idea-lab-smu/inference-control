import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

import learning_rate as lr
from state_history import StateManager

initial_var = 0.0

threshold = 0.006
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
	
		#self.state = []
		var_uni = [1 / 2.0 for i in range(self.size)]
		_, init_state = self.state_manager.compute_dirichlet_stats(var_uni)
		init_history = [0 for i in range(self.size)]
		self.state = init_state # + init_history

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

	
	def is_terminal(self, variances):
	
		res = False

		visit = 0
		valid = 0
		reward = -1
	
		ref = len(variances)

		'''
		Both two conditions should be satisfied
			
			1. the agent should visit all nodes (at least once)
			2. a variance at each node should be smaller than the threshold predefined
		'''
		visit_history = self.state_manager.get_visit_history()
		cur_idx = sum(visit_history) - self.size
		temp_ms = [v[0] for v in self.max_state]

		for i in range(self.size):

			if (visit_history[i] - 1) >= 1:
				visit += 1

			'''
			#if variances[i] <= threshold:
			if self.is_decreasing(self.max_state[i], self.state[i], variances[i]):
				valid += 1
			'''
			if max(temp_ms) >= (threshold) :
				if (variances[i] <= threshold) \
				and (self.is_decreasing(self.state[i], self.new_state[i])):
				
					if (self.max_state[i][0] >= variances[i]) \
						and (self.max_state[i][1] < cur_idx):
						valid += 1	

		#if (visit == self.size) and (valid >= ref):
		#if (min(visit_history) > 3) and (valid >= ref):
		if (visit == self.size) and (valid >= ref):
			
			res = True
			reward = 1

		return res, reward


	# for conjugacy and update rule
	def is_terminal_ext(self, variances):

		valid = 0
		ret = False
		reward = -1

		for v in variances:
			if v <= threshold:
				valid += 1

		history = self.state_manager.get_visit_history()

		if valid == self.size and min(history) > 1:
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

		
	def move_state(self, action):
	
		'''
		steps = self.interpret(action)
		
		new_state = self.state + steps
		
		if new_state > self.size - 1:
			new_state = new_state % self.size
			
		return new_state
		'''
		return action


	def compute_uncertainty(self, action):

		learning_rate, variances = self.state_manager.update(action)

		learning_rate, variances = lr.compute_learning_rate(distribution)

		return learning_rate, variances


	def set_max_state(self):

		idx = sum(self.state_manager.get_visit_history()) - self.size

		for i in range(self.size):

			max_value = max(self.max_state[i][0], self.new_state[i])
			
			self.max_state[i][0]  = max_value
			if max_value == self.new_state[i]:				
				self.max_state[i][1]  = idx

	def merge(self, dest, source, sp, length):

		ret = []

		for sp in range(length):

			dest[sp] = source[sp]			

		ret = dest

		return ret

	def step(self, action):
		
		#new_state = self.move_state(action)

		visits = self.state_manager.get_visit_history()
		#learning_rate, variances = self.compute_uncertainty(action)
		learning_rate, variances = self.state_manager.update(action)		
		new_visits = self.state_manager.get_visit_history()

		self.state = self.new_state
		self.new_state = variances
	
		self.set_max_state()
		self.done, self.reward = self.is_terminal_ext(variances)
		
		return self.new_state, self.reward, self.done, learning_rate
