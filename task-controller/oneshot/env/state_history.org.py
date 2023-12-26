import numpy as np
import math
import dirichlet as dc

class StateManager(object):

	def __init__(self, space_size):

		self.space_size = space_size
		#self.init_prob = [0.0625, 0.0125, 1]
		self.reset()

	def get_buffer(self):

		buf = [[] for s in range(self.space_size)]
		#for i in range(self.space_size):	
		#	buf[i].remove(None)

		return buf

	def reset(self):

		self.visits = [1 for i in range(self.space_size)]

		self.prior = self.get_buffer()
		for i in range(self.space_size):
			posterior = self.update_prior(i)

		self.distribution = self.get_buffer()
		self.update_distribution(posterior)

	def update_visits(self, state):

		self.visits[state] += 1

	def update_prior(self, state):

		'''
		#prob = self.init_prob[state]

		if prob == 1:
			# The probability 1 means it should happen only once.
			# For some reason, it may happen several times 
			# then the prob should be adjusted according to the history. 
			prob = 1 / float(self.visits[state])

		self.prior[state].append(prob)
		'''
	
		#self.prior[state].append(1 / float(self.visits[state]))
		self.prior[state].append(1 / float(3))

		posterior = [np.prod(self.prior[i]) for i in range(self.space_size)]
		
		return posterior	

	def update_distribution(self, posterior):

		total = sum(posterior)

		for i in range(self.space_size):
			self.distribution[i].append(posterior[i] / float(total))

	def get_visit_history(self):

		return self.visits
		
	def update(self, state):

		self.update_visits(state)
		
		#print 'current visit: ', self.visits, '\n'
		
		posterior = self.update_prior(state)
		self.update_distribution(posterior)
		return self.distribution
