import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

'''
This network takes a frame from the game, then flattens into an array.
Next, it resizes the frame and processes it using 4 CONV-2D layers.
'''
class QNetwork():
	
	def __init__(self, input_size, action_size, name):

		self.name = name

		self.input_size = input_size
		self.action_size = action_size
		self.env_actions = [i for i in range(self.action_size)]

		self.hidden_size = 512
		
		# input variables
		self.input = tf.placeholder(shape = [None, self.input_size], dtype = tf.float32)	

		#
		# weight
		#
		self.w_fc_1 = self.init_weights((self.input_size, self.hidden_size))
		self.w_fc_2 = self.init_weights((self.hidden_size, self.hidden_size))
		self.w_fc_3 = self.init_weights((self.hidden_size, self.hidden_size))
		self.w_fc_4 = self.init_weights((self.hidden_size, self.hidden_size))
		self.w_fc_5 = self.init_weights((self.hidden_size, self.hidden_size))		


		#
		# network: feed-forward
		#
		self.h_fc_1 = tf.nn.relu(tf.matmul(self.input, self.w_fc_1))
		self.h_fc_2 = tf.nn.relu(tf.matmul(self.h_fc_1, self.w_fc_2))
		self.h_fc_3 = tf.nn.relu(tf.matmul(self.h_fc_2, self.w_fc_3))
		self.h_fc_4 = tf.nn.relu(tf.matmul(self.h_fc_3, self.w_fc_4))
		self.h_fc_5 = tf.identity(tf.matmul(self.h_fc_4, self.w_fc_5))
	

		# output from the final fully-connected layer 
		# 	-> split into separated advantage and value streams
		self.stream_ac, self.stream_vc = tf.split(self.h_fc_5, 2, 1)
		self.stream_a = slim.flatten(self.stream_ac)
		self.stream_v = slim.flatten(self.stream_vc)
	
		#initializer = tf.contrib.layers.xavier_initializer()
		self.W_a = self.init_weights((self.hidden_size / 2, self.action_size))
		self.W_v = self.init_weights((self.hidden_size / 2, 1))

		self.adv = tf.matmul(self.stream_a, self.W_a)
		self.val = tf.matmul(self.stream_v, self.W_v)

		# towards the final Q-value
		self.r_mean_adv = tf.reduce_mean(self.adv, axis = 1, keep_dims = True)
		self.q_out = self.val + tf.subtract(self.adv, self.r_mean_adv)
		self.predict = tf.argmax(self.q_out, 1)

		# obtain the loss between the target and the predicted Q value
		# (loss = sum of squared difference)
		self.q_target = tf.placeholder(shape = [None], dtype = tf.float32)
		self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
		self.actions_onehot = tf.one_hot(self.actions, self.action_size, dtype = tf.float32)

		self.Q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_onehot), axis = 1)
		
		self.td_err = tf.square(self.q_target - self.Q)
		self.loss = tf.reduce_mean(self.td_err)
		self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
		self.update_model = self.trainer.minimize(self.loss)


	def init_weights(self, shape):

		weight = tf.random_normal(shape, stddev = 0.1)
		return tf.Variable(weight)
















		
		
