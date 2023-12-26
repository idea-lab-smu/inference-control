import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import os

import matplotlib as mpl

from lib.easy_constants import Constant
from lib.easy_experience import ExperienceBuffer

# Load the game env
from env.chainenv import chainEnv
import visualise as vis
import heatmap as hm
import histogram as hs

'''
definition - an optimal seqence that a RL agent generates based upon the neural one-shot model
'''
#sequence = [0, 1, 2, 3, 4, 3, 2, 2, 1, 0, 0, 1, 0, 3, 2, 0, 1, 4, 3, 2]
#sequence = [0, 2, 2, 1, 0, 0, 2, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3] 
#sequence =  [3, 1, 2, 3, 4, 0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 2, 1, 4, 3, 2]

name = 'Max-Oneshot Effect'

# load the hyperparameters and environment variables
const = Constant()
N = const.env_size


'''
def - extract data for visualisation from an episode buffer
return - [list] policy, [list] learning rate, [list] states history
'''
def split_episode(episode, env_size):

	ep_len = len(episode)
	
	states = []
	policy = []
	lr = []

	visits = [0 for i in range(env_size)]

	'''
	A format of episode buffer
	[0]	current state
	[1]	action
	[2]	reward 
	[3]	next state
	[4]	is terminal?
	[5] learnig rate in relation to the current state
	'''

	for i in range(-1, ep_len, 1):
	
		if i < 0:
			action = episode[0][1]
			policy.append('')
			states.append(episode[0][0][:env_size])
			lr.append([0.2 for i in range(5)])
		else:
			action = episode[i][1]
			visits[action] += 1
			policy.append('%d' % episode[i][1])
			states.append(episode[i][3][:env_size])
			lr.append(episode[i][5])

	return policy, lr, states


def simulate_sequence(sequence):

	# load the environment
	env = chainEnv(size = const.env_size)

	# lists for total rewards and steps per each episode
	j_list = []		# number of steps to achieve a goal (the smaller, the better)
	r_list = []		# reward when achieving a goal (the greater, the better)
	history = []

	ep_buf = ExperienceBuffer()

	# reset the env then get the first observation
	s = env.reset()
	terminated = False
	lr = []

	for a in sequence:

		s1, r, terminated, lr = env.step(a)
		ep_buf.add(np.reshape(np.array([s, a, r, s1, terminated, lr]), [1, 6]))
		s = s1

		if terminated == True:
			break

	history.append(env.state_manager.get_visit_history())

	vis.render_episode(ep_buf.buf, const.env_size)	
	return ep_buf.buf


if __name__ == "__main__":
	
	'''
	sequences = [[3, 1, 2, 3, 4, 0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 2, 1, 4, 3, 2],
				[3, 3, 2, 1, 4, 0, 0, 0, 1, 1, 2, 2, 3, 0, 2, 0, 1, 4, 3, 2],
				[0, 3, 2, 1, 4, 1, 1, 0, 2, 2, 3, 0, 3, 2, 0, 0, 1, 4, 2, 3]]
				
	#sequences =  [[3, 1, 2, 3, 4, 0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 2, 1, 4, 3, 2]]
	#[4,4,1,0,3,3,3,0,1,4,1,2,4,1,2,0,2,2,0,3]

	name = 'Max-Oneshot Effect'
	lrs = []
	
	for sequence in sequences:
		episode = simulate_sequence(sequence)
		policy, lr, states = split_episode(episode, N)
		#hs.render(lr)
		lrs.append(lr)
		hm.render(name, policy, lr, states, N, [[0.0, 0.6], [0.0, 0.07], [0.0, 1.0]])
		
	hs.render_ext(lrs)
	'''
	
	sequences =  [[0, 2, 2, 1, 0, 0, 2, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
					[0, 2, 4, 1, 3, 2, 0, 2, 3, 1, 3, 2, 1, 0, 2, 0, 3, 3, 3, 3],
					[4, 1, 0, 2, 3, 1, 2, 0, 2, 3, 1, 3, 1, 2, 0, 0, 3, 0, 3, 3]]
	name = 'Min-Oneshot Effect'	
	lrs = []

	for sequence in sequences:
		episode = simulate_sequence(sequence)
		policy, lr, states = split_episode(episode, N)
		#hs.render(lr)
		lrs.append(lr)
		hm.render(name, policy, lr, states, N, [[0.0, 0.6], [0.0, 0.07], [0.0, 1.0]])
		
	hs.render_ext(lrs)

