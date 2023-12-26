class Constant():

	def __init__(self):

		# a number of nodes in the environment
		self.env_size = 5

		# the objective of this ddqn - max, min or no
		self.case = 'max-oneshot'

		# the number of experiences used for each training step
		self.batch_size = 32

		# frequency to perform a training step
		self.update_frequency = 4

		# discount factor
		self.gamma = 0.99

		# starting chance of the random action, S_E
		self.p_rand_start = 1	

		# final chance of the random action, E_E
		self.p_rand_end = 0.1

		# 
		self.annealing_steps = 10000.

		#
		self.n_ep = 100000

		#
		self.pretrain_steps = 500000
		#self.pretrain_steps = 12500
		
		#
		self.max_ep_len = 100
		#self.max_ep_len = 50

		# the flag representing whether to load the saved model
		self.load_model = True

		#
		self.path_model = './models/ddqn-%s-node-%s-2.0/' % (self.env_size, self.case)

		# the size of the final conv layer
		# (before splitting into advantage and value streams)
		self.H = 512

		# target update rate toward the primary network
		self.tau = 0.001
