'''
  asa266 - Dirichlet Core

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
import asa266_utils as au 

class Dirichlet(object):
   
	def __init__(self, sample):

		''' 
			Initialise dirichlet object 
		'''
		try:
			nparam = len(sample)
			if nparam < 2:
				raise Exception('Dirichlet Estimate Error', \
								'Parameters should be greater than 2')

			nelem = len(sample[0])
			if nelem < 1:
				raise Exception('Dirichlet Estimate Error', \
								'Elements in the sample should have at leat one item')

			self.sample = sample
			self.row = nparam
			self.col = nelem
			self.alpha = []
			self.v = []
			self.g = []

		except Exception as inst:
			err, desc = inst.args
			print err, ': ', desc

	'''
		initial estimation using the method of moments
	'''
	def _estimate_by_mom(self, sample):

		# calculate initial estimate using the method of moments (mem)

		alpha = []
		row = len(sample)
		col = len(sample[0])

		for i  in range(row - 1):
			alpha.insert(i, sum(sample[i]) / float(col))
		alpha.insert(row - 1, 1.0 - sum(alpha))
		
		s12 = 0.0
		for i in range(col):
			s12 = s12 + pow(sample[0][i], 2)

		s12 = s12 / float(col)
		varp1 = s12 - pow(alpha[0], 2)

		s11 = (alpha[0] - s12) / varp1
		for i in range(row):
			alpha[i] = s11 * alpha[i]

		return alpha

	'''
		initial estimation using Ronning's suggestion
	'''
	def _estimate_by_ronning(self, sample):

		# calcualte initial estimate using ronning's suggestion
		print '__estimate_by_ronning__: not implemented'

	'''
		check whether alpha is a negative value
	'''
	def _is_positive(self, alpha):
		
		if min(alpha) < 0:
			return False

		return True

	'''
		calculate variance on alpha estimation

	def _alpha_variance(self, v, row, col):
		
		vari = []
		for i in range(row):
			tv = v[i + i * row]
	'''

	''' 
		Dirichlet estimate 

		Return:
			alpha - output, a list of dirichlet estimates

		Argument:
			init - input, specifies how the parameter estimates are to be initialised:
				1. use the method of mements
				2. initialise each ALPHA to the minimum of X; 
				otherwise, assume that the input calues of ALPHA already contain estimates	
	'''
	#def estimate(self, init, llh_value, cov, g, niter, chi_value, eps):
	def estimate(self, init):

		gamma = 0.0001
		alpha_min = 0.00001
		it_max = 100
		rlogl = 0.0

		sample = self.sample
		row = self.row
		col = self.col
		alpha = self.alpha
		v = self.v
		g = []

		f_row = float(row)
		f_col = float(col)

		''' initial estimation '''
		if init == 1:
			# using the method of moments
			alpha = self._estimate_by_mom(sample) 
		elif init == 2:
			# using Ronning's suggestion
			alpha = self._estimate_by_ronning(sample)
		else:
			# exception
			raise Exception('Dirichlet Estimate Error', 'Specify init either 1 or 2') 

		if self._is_positive(alpha) is False:
			raise Exception('Dirichlet Estimate Error', 'Alpha must be positive')

		work = [] # length = number of parameters that will be estimated
		for i in range(row):
			s_log = 0.0
			for j in range(col):
				s_log = s_log + math.log(sample[i][j])
			work.insert(i, s_log)

		# Call algorithm AS 91 to compute CHI2, the chi-squared value
		gg = math.lgamma(f_row / 2.0)
		chi2 = au.ppchi2(gamma, f_row, gg)

		# Carry out the newton iteration
		work2 = []
		for it in range(1, it_max + 1):
			sum2 = sum(alpha) # change the custom routine to api
			sum1 = 0.0

			for i in range(row):
				work2.insert(i, au.trigamma(alpha[i]))
				sum1 = sum1 + 1.0 / work2[i]
			
			beta = au.trigamma(sum2)
			beta = f_col * beta / (1.0 - beta * sum1)
			
			temp = au.digamma(sum2)

			for i in range(row):
				res = au.digamma(alpha[i])
				g_value = f_col * (temp - res) + work[i]
				g.insert(i, g_value)

			# Calculate the lower triangle of the Variance-Covariance matrix V.
			sum2 = beta / f_col / f_col

			v = range(row * row)
			for i in range(row):
				for j in range(row):
					index = i + j * row
					v[index] = sum2 / (work2[i] * work2[j])
					if i == j:
						vtemp = v[index]
						v[index] = vtemp + 1.0 / (f_col * work2[j])

			# post-multiply the Variance-Covarinace matrix V by G and store in work2
			work2 = au.r8mat_mv_new(row, row, v, g)

			# update the ALPHA'S
			for i in range(row):
				alpha[i] = alpha[i] + work2[i]
				alpha[i] = max(alpha[i], alpha_min)
			
			# update the variance on ALPHA
			#var_alpha = _alpha_variance(v, row, col)
			self.alpha = alpha
			#self.var_alpha = var_alpha

			# test for convergence
			s = au.r8vec_dot_product(row, g, work2)

			if s < chi2:
				eps = au.gammad(s / 2.0, f_row / 2.0)

				sum2 = sum(alpha)
				
				rlogl = 0.0
				for i in range(row):
					rlogl = rlogl + (alpha[i] - 1.0) * work[i] - f_col * math.lgamma(alpha[i])

				rlogl = rlogl + f_col * math.lgamma(sum2)

				return alpha, v, rlogl

		return alpha, v, rlogl

	'''
		Dirichlet_mean
	'''


	'''
		dirichlet_variance
	'''

