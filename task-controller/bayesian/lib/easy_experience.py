import numpy as np
import random

class ExperienceBuffer():

	def __init__(self, buf_size = 50000):

		self.buf = []
		self.buf_size = buf_size

	def add(self, experience):
		
		size_expected = len(self.buf) + len(experience)

		# perhaps overflow?
		if (size_expected >= self.buf_size):
			# prepare the buf to extend the original buf
			self.buf[0:size_expected - self.buf_size] = []

		self.buf.extend(experience)

	def sample(self, size):
		
		return np.reshape(np.array(random.sample(self.buf, size)), [size, 6])



