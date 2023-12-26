'''
Q-network for Co-Evolution Project 

	The design of the network is inspired by https://github.com/spiglerg/

	jeehanglee@gmail.com
	28 Jun 2017
'''

import numpy as np
import random
import math

import tensorflow as tf

'''

Base class for Q-Network

'''

class QNetwork(object):

	def __init__(self, input_size, output_size, name):

		self.name = name


	def weight_variables(self, shape, fanin = 0):

		if fanin == 0:

			initial = tf.truncated_normal(shape, stddev = 0.01)

		else:

			mod_init = 1.0 / math.sqrt(fanin)
			initial = tf.random_uniform(shape, minval = -mod_init, maxval = mod_init)

		return tf.Variable(initial)


	def bias_variables(self, shape, fanin = 0):

		if fanin == 0:

			initial = tf.constant(0.01, shape = shape)

		else:
		
			mod_init = 1.0 / math.sqrt(fanin)
			initial = tf.random_uniform(shape, minval = -mod_init, maxval = mod_init)

		return initial


	def variables(self):

		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
		return variables


	def copy_to(self, target_net):

		'''
		mn = ModelNetwork(2, 3, 0, "actor")
		mn_target = ModelNetwork(2, 3, 0, "target_actor")
		
		s=tf.InteractiveSession()

		s.run( tf.initialize_all_variables() )

		mn.copy_to(mn_target)
		'''

		v_s = self.variables()			# source network variables
		v_t = target_net.variables()	# target network variables

		for i in range(len(v_s)):

			v_t[i].assign(v_s[i]).eval()


	def __print_parameters__(self):

		list_vars = self.variables()
		total_parameters = 0

		for variable in list_vars:

			# shape is an array of tf.Dimension
			shape = variable.get_shape()
			variable_parametes = 1

			for dim in shape:

				variable_parametes *= dim.value

			total_parameters += variable_parametes

		print '# of parameters in the network ', self.name, ': ', \
				total_parameters, '  ->  ', \
				np.round(float(total_parameters)/1000000.0, 2),'M'


'''

Implementation of QNetwork

'''
class QNetwork_BML(QNetwork):

	"""
	QNetwork used in [some paper], [JeeHang et al., 201?].

	It's a Fully Connected Neural Network with the following specs:
		L1: 10 * input_size Fully-Connected layer 			RELU
		L2: 2 * 10 * input_size Fully-Connected Layer		RELU
		L3: 2* 10 * input_size Fully-Connected layer		RELU
		L4: 10 * input_size Fully-Connected layer			RELU
		L5: [output_size] output units, Fully Connected		Softmax
	"""


	def __init__(self, input_size, output_size, name):

		self.name = name

		self.input_size = input_size[0]
		self.output_size = output_size

		# Build network
		with tf.variable_scope(self.name):

			self.w_fc_1 = self.weight_variables([self.input_size, self.input_size * 10])
			self.b_fc_1 = self.bias_variables([self.input_size * 10])

			self.w_fc_2 = self.weight_variables([self.input_size * 10, self.input_size * 10 * 2])
			self.b_fc_2 = self.bias_variables([self.input_size * 10 * 2])

			self.w_fc_3 = self.weight_variables([self.input_size * 10 * 2, self.input_size * 10 * 2])
			self.b_fc_3 = self.bias_variables([self.input_size * 10 * 2])

			self.w_fc_4 = self.weight_variables([self.input_size * 10 * 2, self.input_size * 10])
			self.b_fc_4 = self.bias_variables([self.input_size * 10])

			self.w_out = self.weight_variables([self.input_size * 10, self.output_size])
			self.b_out = self.bias_variables([self.output_size])

		# print number of parameters in the network
		self.__print_parameters__()


	def __call__(self, input_tensor):

		if type(input_tensor) == list:

			input_tensor = tf.concat(1, input_tensor)

		with tf.variable_scope(self.name):

			# input tensor is (input_size)

			self.h_fc_1 = tf.nn.relu(tf.matmul(input_tensor, self.w_fc_1) + self.b_fc_1)

			self.h_fc_2 = tf.nn.relu(tf.matmul(self.h_fc_1, self.w_fc_2) + self.b_fc_2)

			self.h_fc_3 = tf.nn.relu(tf.matmul(self.h_fc_2, self.w_fc_3) + self.b_fc_3)

			self.h_fc_4 = tf.nn.relu(tf.matmul(self.h_fc_3, self.w_fc_4) + self.b_fc_4)

			self.out = tf.identity(tf.matmul(self.h_fc_4, self.w_out) + self.b_out)

		return self.out
		
