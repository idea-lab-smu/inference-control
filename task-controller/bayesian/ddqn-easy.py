import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import os

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf

from lib.easy_constants import Constant
from lib.easy_qnetwork import QNetwork
from lib.easy_experience import ExperienceBuffer

# Load the game env
from env.chainenv import chainEnv

#import viz_3d_tools as viz
import visualise as vis

# update parameters of the target network 
# with those of the primary network
def update_target_graph(tf_vars, tau):
	
	vars_size = len(tf_vars)
	op_holder = []

	for idx, var in enumerate(tf_vars[0:vars_size // 2]):

		val_prev = tf_vars[idx + vars_size // 2].value()
		val_next = var.value() * tau + (1 - tau) * val_prev
		op = tf_vars[idx + vars_size // 2].assign(val_next)
		op_holder.append(op)

	return op_holder

def update_target(op_holder, sess):

	for op in op_holder:
		sess.run(op)


# load the hyperparameters and environment variables
const = Constant()


# load the environment
env = chainEnv(size = const.env_size)

'''
Training the Network
'''


tf.reset_default_graph()
main = QNetwork(const.env_size, const.env_size, 'online Q network')
target = QNetwork(const.env_size, const.env_size, 'target Q network')

init = tf.global_variables_initializer()
saver = tf.train.Saver()
buf = ExperienceBuffer()

trainables = tf.trainable_variables()
target_ops = update_target_graph(trainables, const.tau)

# set the rate that the random action decreases
e = const.p_rand_start
step_drop = (const.p_rand_start - const.p_rand_end) / const.annealing_steps

# lists for total rewards and steps per each episode
j_list = []		# number of steps to achieve a goal (the smaller, the better)
r_list = []		# reward when achieving a goal (the greater, the better)
policy = []
history = []
n_steps = 0

# the file path to save the model
if not os.path.exists(const.path_model):
	os.makedirs(const.path_model)

# start!
with tf.Session() as sess:

	sess.run(init)

	if const.load_model == True:

		print 'Load the model already saved...'
		check_point = tf.train.get_checkpoint_state(const.path_model)
		saver.restore(sess, check_point.model_checkpoint_path)

	update_target(target_ops, sess)

	for i in range(const.n_ep):

		ep_buf = ExperienceBuffer()
		
		# reset the env then get the first observation
		s = env.reset()
		terminated = False
		r_all = 0
		j = 0
		actions = []
		lr = []

		# The Q-Network
		# we will terminate the trial if the agent shows more than 200 moves
		while j < const.max_ep_len:

			j += 1
			
			'''
			if const.load_model == True:

				a = sess.run(main.predict, feed_dict = {main.input:[s]})[0]

			else:
			'''
			
			# choose the action from the Q-Network
			if np.random.rand(1) < e or n_steps < const.pretrain_steps:
				a = np.random.randint(0, const.env_size)
			else:
				a = sess.run(main.predict, feed_dict = {main.input:[s]})[0]

			s1, r, terminated, lr = env.step(a)
			n_steps += 1
			ep_buf.add(np.reshape(np.array([s, a, r, s1, terminated, lr]), [1, 6]))

			if n_steps > const.pretrain_steps:

				if e > const.p_rand_end:

					e -= step_drop

				if n_steps % const.update_frequency == 0:

					# get a random batch of experiences
					train_batch = buf.sample(const.batch_size)
					
					# double dqn update to the target Q-value
					q1 = sess.run(main.predict, feed_dict = {main.input : np.vstack(train_batch[:,3])})
					q2 = sess.run(target.q_out, feed_dict = {target.input : np.vstack(train_batch[:,3])})

					end_mul = -(train_batch[:,4] - 1)
					double_q = q2[range(const.batch_size), q1]
					target_q = train_batch[:,2] + (const.gamma * double_q * end_mul)

					# update the network with the target values
					_ = sess.run(main.update_model, \
								feed_dict = {main.input : np.vstack(train_batch[:,0]), \
											main.q_target : target_q, \
											main.actions:train_batch[:,1]})

					update_target(target_ops, sess)

			r_all += r
			s = s1
			actions.append(a)

			if terminated == True:
				break

		buf.add(ep_buf.buf)
		j_list.append(j)
		r_list.append(r_all)
		policy.append(actions)
		history.append(env.state_manager.get_visit_history())

		# for a test log
		if r_all > -100:
			print '\tReward: ', r_all, \
				' in ', j, 'steps ', \
				', history: ', env.state_manager.get_visit_history()

			print '\tfinal state ', s[:3]

		# save the model in periodic
		if i % 1000 == 0 and i > 0:
			saver.save(sess, const.path_model + '/model-' + str(i) + 'cptk')
			print 'the model saved...'

		if len(r_list) % 10 == 0:
			print '***'
			print n_steps, np.mean(r_list[-10:]), e, len(r_list), i
			print '***'

		if const.load_model == True:
			if n_steps > const.pretrain_steps * 1.5 and r_all == -18:
				vis.render_episode(ep_buf.buf, const.env_size)

	saver.save(sess, const.path_model + '/model-' + str(i) + 'cptk')

print 'Percentage of successful episodes: ' + str(sum(r_list) / const.n_ep) + '%'

plt.figure()
r_mat = np.resize(np.array(r_list), [len(r_list) // 100, 100])
r_mean = np.average(r_mat, 1)
plt.plot(r_mean)

plt.figure()
j_mat = np.resize(np.array(j_list), [len(j_list) // 100, 100])
j_mean = np.average(j_mat, 1)
plt.plot(j_mean)

plt.figure()
avg_steps = [x / const.env_size for x in j_mean]
plt.plot(avg_steps)
plt.show()

#viz.scatter_3d(history)

print 'end of program'
			
