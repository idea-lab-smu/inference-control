import numpy as np
import math
import copy

from env.chainenv import chainEnv
import visualise as vis

'''
definition
'''

def render(qvalue):
	
	size = len(qvalue)
	buf = [[0] * size for s in range(size)]

	for i in range(size):

		pos = np.argmax(qvalue[i,:])
		
		'''
		if pos >= i:
			pos += 1
		'''

		buf[i].insert(pos, 1)
		print buf[i], '\n'

'''
Environment Configuration
'''
ENV_SIZE = 3
ACT_SIZE = ENV_SIZE

'''
program entry point
'''

# Load the environment
env = chainEnv(size = ENV_SIZE) 

# initialise the table with all zeros
Q = np.zeros([ENV_SIZE, ACT_SIZE])
Q_OLD = np.zeros([ENV_SIZE, ACT_SIZE])

# Set learning (hyper) parameters
learning_rate = 0.9
gamma = 0.99
max_ep = 10

# create lists containing total rewards and steps per episode
r_list = []
visit = []
q_list = []
ep_buf = []
pv_buf = []	# a list of posterior variance

for i in range(max_ep):

	# reset the env and get first new onbservation
	# s = env.reset([10, 10, 1])
	s = env.reset()
	tot_rewards = 0
	d = False
	j = 0
	episode = []
	pv = []

	# Q-table learning algorithm
	while j < 100:
		j += 1

		# choose the action by e-greedy (with noise) picking from Q table
		# action = np.argmax(Q[s,:] + np.random.randn(1, ACT_SIZE) * (1./ (i + 1)))
		action = np.argmax(Q[s,:])

		# get new state and reward from the environment
		ns, rewards, pos_var = env.step(action)
		v_buf = [v for v in env.state_manager.visits]
		episode.append([s, action, rewards, ns, v_buf])
		pv.append(pos_var)
		
		# update q-table with new knowledge

		Q_OLD = copy.copy(Q)

		# Q[s, action] = Q[s, action] + learning_rate * (reward + gamma * np.max(Q[ns,:]) - Q[s, action])
		#index = np.argmax(rewards)
		#Q[s, index] = Q[s, index] + learning_rate * (max(rewards) + gamma * np.max(Q[ns,:]) - Q[s, index])
		for i in range(len(rewards)):
			Q[s, i] = Q_OLD[s, i] + learning_rate * (rewards[i] + gamma * np.max(Q_OLD[ns,:]) - Q_OLD[s, i])

		tot_rewards += max(rewards)
		s = ns

		# is terminal state?
		# if d == True:
		#	break

	r_list.append(tot_rewards)
	ep_buf.append(episode)
	visit.append(env.state_manager.visits)
	pv_buf.append(pv)
	
	render(Q)
	q_list.append(Q)

	print '# of visits: ', env.state_manager.visits
	print Q

print 'Score over time: ', sum(r_list) / max_ep   
print 'Final Q-Table value: ', '\n', Q

vis.draw(r_list, ep_buf, visit, q_list, pv_buf)
