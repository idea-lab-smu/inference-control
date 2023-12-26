'''
  asa266 Test Code -  Dirichlet 

  Purpose:

    DIRICHLET_ESTIMATE, DIRICHLET_MEAN, DIRICHLET_VARIANCE.

  Discussion:

    Canned data is used.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    05 June 2013
    29 Mar  2017

  Author:

    Original C version by John Burkardt
    Python version by JeeHang Lee (jeehanglee@gmail.com)
'''

import numpy as np
import math

from asa266.asa266_dirichlet import Dirichlet

# Definition
LF = '\n'
ID = '  '

'''
'''
def print_matrix(mat, s_fmt):
	
	sample_t = map(list, zip(*sample))
	for s in sample_t:
		pstr = ''
		l = len(s)
		for i in range(l):
			pstr += s_fmt % s[i]
		print pstr
	print LF

'''	
#
# samples - for test
#
sample = [[0.178, 0.162, 0.083, 0.087, 0.078, 0.040, 0.049, 0.100, 0.075, 0.084,
    0.060, 0.089, 0.050, 0.073, 0.064, 0.085, 0.094, 0.014, 0.060, 0.031,
    0.025, 0.045, 0.0195],
    [0.346, 0.307, 0.448, 0.474, 0.503, 0.456, 0.363, 0.317, 0.394, 0.445,
    0.435, 0.418, 0.485, 0.378, 0.562, 0.465, 0.388, 0.449, 0.544, 0.569,
    0.491, 0.613, 0.526],
    [0.476, 0.531, 0.469, 0.439, 0.419, 0.504, 0.588, 0.583, 0.531, 0.471,
    0.505, 0.493, 0.465, 0.549, 0.374, 0.450, 0.518, 0.537, 0.396, 0.400,
    0.484, 0.342, 0.4545]]
'''
	
'''
compute learning rate per each state an agent will newly visit
'''
def compute_learning_rate(sample):

	#sample = [s[1:] for s in sample]
	
	#
	# compute the observed mean and variance
	#
	for i in range(len(sample)):
		obs_mean = np.mean(sample[i])
		obs_var = np.var(sample[i])

	#
	# Dirichlet estimation
	#
	drcl = Dirichlet(sample)

	alpha, v, rlogl = drcl.estimate(1)
	alpha_sum = sum(alpha)
	alpha_mean = []
	alpha_var = []

	#
	# estimate, Lower and Upper limit:
	#
	row = len(sample)
	for i in range(row):
		vari = v[i + i * row]
		aminus = (alpha[i] - 1.96 * math.sqrt(vari)) 
		aplus = (alpha[i] + 1.96 * math.sqrt(vari)) 

	#
	# Dirichlet mean and variance
	#
	for i in range(row):
		mean = alpha[i] / alpha_sum
		alpha_mean.insert(i, mean)

		variance = (alpha[i] * (alpha_sum - alpha[i])) / (alpha_sum * alpha_sum * (alpha_sum + 1.0))
		alpha_var.insert(i, variance)

	#print sample, '\n'
	#print alpha, '\n'
	#print alpha_var, '\n'

	#
	# Normalised values
	#
	row = len(sample)
	for i in range(row):
		vari = v[i + i * row]
		aminus = (alpha[i] - 1.96 * math.sqrt(vari)) / alpha_sum
		aplus = (alpha[i] + 1.96 * math.sqrt(vari))  / alpha_sum

	#
	# Learning Rate (to evaluate the one-shot learning effect)
	#
	#	Reference
	#		Neural computations mediating one-shot learning in the human brain
	#		Lee et al., PLOS Biology, 2015
	#

	#print ID, 'ONE-SHOT LEARNING MODEL'
	#print ID, 'Learning rate for each stimulus, the posterior'
	tau = 100 # amplify 100 times
	lrate = []
	avsum = 0
	i = 0

	for av in alpha_var:
		avsum = avsum + math.exp(tau * av)

	for av in alpha_var:
		lr = math.exp(tau * av) / avsum
		lrate.insert(i, lr)

		posterior = lr * alpha[i]
		if max(alpha) != alpha[i]:
			posterior = -posterior

		#print '%6d%14.8f%14.8f' % (i, lr, posterior)
		i += 1
	#print LF

	#print ID, 'Sum of exponential variance is %.8f' % avsum
	#print ID, 'Inverse temperature parameter is %d' % tau, LF

	return lrate, alpha_var

