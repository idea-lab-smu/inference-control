import matplotlib.pyplot as plt
import matplotlib.style as st
import numpy as np
import pandas as pd
import math

import heatmap as hm

def save(r_list, sequence, visit, q_list):

	reward = pd.DataFrame(r_list, columns = ['Total Reward'])
	reward.to_csv('reward.csv')

	full_data = pd.DataFrame(sequence)
	full_data.to_csv('full.csv')

	visit_data = pd.DataFrame(visit)
	visit_data.to_csv('visit.csv')
	
	'''
	q_data = pd.DataFrame(q_list)
	q_data.to_csv('q.csv')
	'''

def draw_box(visit):
	
	# draw box
	t_visit = np.asarray(visit).T
	plt.figure()
	plt.boxplot(t_visit.tolist())
	#plt.show()
	
def draw_visit(visit):

	n_steps = sum(visit[0]) - len(visit[0])
	n_ep = len(visit)

	text = 'State %d'
	col = [(text % (i + 1)) for i in range(len(visit))]

	#df_visit = pd.DataFrame(visit, col)
	df_visit = pd.DataFrame(visit)
	ax = df_visit.plot.bar(stacked = True)
	ax.set_xlim(0, n_ep)
	ax.set_ylim(0, n_steps)
	#plt.show()
		
def draw_reward(r_list):

	ax = plt.plot(r_list, label = 'Total reward of each episode')

	maximum = max(r_list)
	minimum = min(r_list)
	
	plt.axis([0, len(r_list), minimum - 10, maximum + 10])
	plt.legend()
	#plt.show()
		
def draw_episode(sequence, r_list, visit):

	print 'The argmax is ', np.argmax(r_list), '\n'
	print 'The argmin is ', np.argmin(r_list), '\n'
	
	r_old = 0

	high = np.argmax(r_list)
	low = np.argmin(r_list)

	n_steps = sum(visit[0]) - len(visit[0])
	n_ep = len(visit)
	n_state = len(visit[0])

	for s in range(len(sequence)):

		if s != high and s != low:
			continue
	
#		if r_old == r_list[s]:
#			continue

		data = []
		ticks = []

		for i in range(n_steps):
		
			data.append(sequence[s][i][2])
			ticks.append(str(sequence[s][i][4]))
			
		t_data = np.asarray(data).T
		for i in range(len(t_data)):
			plt.plot(t_data[i], label = 'State %d' % i)
		
		x = np.array([i for i in range(n_steps)])
		plt.xticks(x, ticks, rotation='vertical')
		
		plt.legend()
		#plt.show()
		
		r_old = r_list[s]

def draw_ep_stats(ep_buf, visit):

	mean_buf = []
	std_buf = []
	
	n_steps = sum(visit[0]) - len(visit[0])
	n_ep = len(visit)
	n_state = len(visit[0])

	for s in range(n_ep):
		
		mean = []
		std = []
		reward = []
		
		for i in range(n_steps):
			reward.append(ep_buf[s][i][2])
		
		t_data = np.asarray(reward).T
		
		for j in range(len(t_data)):
			mean.append(t_data[j].mean())
			std.append(t_data[j].std())
			
		mean_buf.append(mean)
		std_buf.append(std)
	
	#plt.plot(mean_buf)
	#plt.show()
	
	t_mean = np.asarray(mean_buf).T
	t_std = np.asarray(std_buf).T
	
	rsrc = 'State %d'	
	col = [(rsrc % (i + 1)) for i in range(n_state)]
	index = [i for i in range(n_ep)]

	plt.figure()
	
	for k in range(n_state):

		mf = t_mean[k]
		sf = t_std[k]	

		plt.plot(index, mf, 'C%d' % k, alpha = 0.8, label = col[k])
		plt.fill_between(index, mf - 0.5 * sf, mf + 0.5 * sf, color = 'C%d' % k, alpha = 0.1)
	
	plt.axis([0, n_ep, 0, 1])
	plt.legend()
	#plt.show()

def draw(r_list, sequence, visit, q_list, pv_buf):

	st.use('ggplot')

	save(r_list, sequence, visit, q_list)

	'''
	draw_reward(r_list)
	draw_visit(visit)
	draw_box(visit)
	draw_ep_stats(sequence, visit)
	'''
	draw_episode(sequence, r_list, visit)

	plt.show()	


def compute_learning_rate(alpha_var):

	tau = 100 # amplify 100 times
	lrate = []
	avsum = 0
	i = 0

	for av in alpha_var:
		avsum = avsum + math.exp(tau * av)

	for av in alpha_var:
		lr = math.exp(tau * av) / float(avsum)
		lrate.append(lr)

	return lrate


def render_episode(episode, env_size):

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
	#states.append(episode[0][0][:env_size])
	#lr.append(compute_learning_rate(episode[0][0][:env_size]))	

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

	#visits = episode[ep_len - 1][3][env_size:]
	#hm.render('', policy, lr, states, env_size, [[0.0, 0.6], [0.0, 0.07], [0.0, 1.0]])
	
	# rendering the plots
	x = np.arange(0, len(policy) + 1, 1)
	
	#
	# first figure - uncertainty over policy
	#	
	fig = plt.figure(figsize = (6, 2.5))
	#fig.suptitle('Visit history = ' + str(visits))
	
	#ax = fig.add_subplot(212)
	#ax.set_title('Uncertainty on each S-O pair')
	plt.title('Uncertainty on each node pair')

	plots = plt.plot(states, marker='.')

	plt.ylim(0.0, 0.15)
	plt.xlabel('Policy')
	plt.xticks(x, policy)
	#plt.ylabel('posterior varinace - uncertainty')
	#plt.yticks(y, yticks)

	legend_txt = ['S' + str(i + 1) for i in range(len(plots))]
	plt.legend(plots, legend_txt, loc = 2)

	'''
	#
	# second figure - learning rate over policy
	#
	ax2 = fig.add_subplot(211)
	ax2.set_title('Learning Rate')
	
	plots = plt.plot(lr, marker = '.')

	plt.ylim(0.0, 0.8)
	plt.xticks(x, policy)
	#plt.ylabel('Learning Rate')
	# plt.yticks(y, yticks)

	legend_txt = ['S' + str(i + 1) for i in range(len(plots))]
	plt.legend(plots, legend_txt, loc = 2)
	'''

	plt.subplots_adjust(hspace = 0.25)
	plt.tight_layout()
	plt.show()




