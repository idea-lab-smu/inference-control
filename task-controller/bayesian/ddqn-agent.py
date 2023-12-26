import time
import logging
import shutil
import os
import sys

import tensorflow as tf

# garbage collector
import gc
gc.enable()

# Custom packages
from lib.dqn import *
#from lib.env_utils import *

# For environmental setting for DDQN agents
from env.chainenv import chainEnv

import matplotlib.pyplot as plt

'''
add packages
'''

print '-------------------------------------------------------'
print "Usage:"
print "  ",sys.argv[0]," [optional: path_to_ckpt_file] [optional: True/False test mode]"
print '-------------------------------------------------------'
print

outdir = 'result'

ENV_NAME = 'Markov-Chain-Toy'
ENV_SIZE = 3


'''
1 GAME <= MAX_TRAINING_STEPS
1 EPOCH == 10 GAMES
'''

# max steps before resetting the environment
MAX_TRAINING_STEPS = 100			

# one epoch consists of 10 games that each of game has 100 steps
ONE_EPOCH_SIZE = 10 * 100

# so we have 50000 games to do in total
MAX_TOTAL_STEPS = 500 * ONE_EPOCH_SIZE 	

#
MAX_TESTING_GAMES = 10

#
MAX_TESTING_STEPS = 100


# set log directory
LOG_DIR = outdir + '/' + ENV_NAME + '/logs/'
if os.path.isdir(LOG_DIR):
	shutil.rmtree(LOG_DIR)
journalist = tf.summary.FileWriter(LOG_DIR)


# Build Environment
env = chainEnv(size = ENV_SIZE)
env_shape = np.array([i for i in range(ENV_SIZE)]).shape


# Initialise a TensorFlow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config = config)


# Create DDQN agent
useDoubleDQN = True

agent = DQN(state_size = env_shape,
				action_size = ENV_SIZE,
				session = session,
				summary_writer = journalist,
				exploration_period = ONE_EPOCH_SIZE,
				minibatch_size = 8,
				discount_factor = 0.99,
				experience_replay_buffer = 512 * 512,
				target_qnet_update_frequency = 100, #30000 if UseDoubleDQN else 10000, ## Tuned DDQN
				initial_exploration_epsilon = 1.0,
				final_exploration_epsilon = 0.1,
				reward_clipping = 1.0,
				DoubleDQN = useDoubleDQN)

session.run(tf.global_variables_initializer())
journalist.add_graph(session.graph)

saver = tf.train.Saver(tf.global_variables())
logger = logging.getLogger()
logging.disable(logging.INFO)


# If an argument is supplied, load the specific checkpoint
test_mode = False

if len(sys.argv) >= 2:

	saver.restore(session, sys.argv[1])

if len(sys.argv) == 3:

	test_mode = sys.argv[2] == 'True'



# 
#
#

# indicating a total number of steps in this whole loop
num_steps = 0

# indicating how many games have been performed so far
num_games = 0

# indicating how many steps have been taken since the current game started
current_game_steps = 0

# 
last_time = time.time()
last_step_count = 0.0


# reset the environment
state = env.reset()


'''
1 GAME <= MAX_TRAINING_STEPS = 100
1 EPOCH == 100 GAMES

MAX_TRAINING_STEPS = 100			# max steps before resetting the environment
MAX_EPOCH_SIZE = 100				# number of games (or episodes)
MAX_TOTAL_STEPS = 400 * MAX_EPOCH_SIZE	# number of iterations
'''
# start the iteration
#  - The total number of iterations means MAX_EPOCH_SIZE games are played. 
#  - The number of steps per each game is MAX_TRAINING_STEPS. 
#  - We initially want to run 40000 games

reward_buf = []
step_buf = []

while num_steps <= MAX_TOTAL_STEPS + 1:

	num_steps += 1
	current_game_steps += 1	

	if not test_mode:
		action = agent.action(state, training = True)
	else:	
		action = agent.action(state, training = False)

	
	# Perform the selected action on the environment
	next_state, reward, done = env.step(action)

	
	# Store experience  
	# --  should be checked 20170711 experience in this case should contain a posterior variance
	agent.store(state, action, reward, next_state, done)
	state = next_state

	
	# Train an agent after 10 games carried out
	if num_steps >= ONE_EPOCH_SIZE:
	
		agent.train()


	# We do update variables when 
	#	1. the previous action leads the game terminated 
	#	(i.e. the termination condition is satisfied, the goal is achieved for example)
	# 	2. a RL agent takes MAX_TRAINING_STEPS although the goal is not achieved yet
	if done or current_game_steps > MAX_TRAINING_STEPS:

		state = env.reset()
		current_game_steps = 0
		num_games += 1

	# Print an update every 10 games
	if num_steps % ONE_EPOCH_SIZE == 0:

		new_time = time.time()
		diff = new_time - last_time
		last_time = new_time

		elapsed_steps = num_steps - last_step_count
		last_step_count = num_steps

		print 'Steps: ', num_steps, '\tGames: ', num_games, '\tSpeed: ', (elapsed_steps/diff), 'steps/sec'


	# save the network parameters after every epoch
	if num_steps % ONE_EPOCH_SIZE == 0 and num_steps > ONE_EPOCH_SIZE:
		
		saver.save(session, outdir + "/" + ENV_NAME + "/model_" + str(num_steps/1000)+"k.ckpt")

		print
		print "epoch:  steps = ", num_steps, "\tgames = ", num_games

	
	# Testing - it's kind of slow, so we're only going to test every 2 epochs
	
	if num_steps % (2 * ONE_EPOCH_SIZE) == 0 and num_steps > ONE_EPOCH_SIZE:

		total_reward = 0
		avg_steps = 0

		for i in xrange(MAX_TESTING_GAMES):

			state = env.reset()
			steps = 0

			while steps < MAX_TESTING_STEPS:

				steps += 1
				action = agent.action(state, training = False) # direct action for the test
				
				state, reward, done = env.step(action)

				total_reward += reward

				if done:
					break

			avg_steps += steps

		avg_reward = float(total_reward) / MAX_TESTING_GAMES

		_str_ = session.run(tf.summary.scalar('test reward (' + str(ONE_EPOCH_SIZE/1000) + 'k)', avg_reward))
		journalist.add_summary(_str_, num_steps)

		print '\t--> Evaluation Average Reward: ', avg_reward, '\tavg steps: ', (avg_steps / MAX_TESTING_GAMES)

		reward_buf.append(avg_reward)
		step_buf.append(avg_steps / MAX_TESTING_GAMES)		

		state = env.reset()

journalist.close()

plt.plot(reward_buf)
plt.plot(step_buf)
plt.show()

# Save the final network
saver.save(session, outdir + "/" + ENV_NAME + "/final.ckpt")











