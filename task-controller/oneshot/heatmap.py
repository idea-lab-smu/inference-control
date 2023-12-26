import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


'''
Strings for titles
'''
suptitle = 'Optimised Knowledge Exploration of an Oneshot RL Agent'
title_seq = 'Sequence'
title_lr = 'Learning Rate'
title_s = 'Uncertainty on each S-O pair'

'''
heatmap configurations regarding the env size.
-- For example, when env_size == 5, the total steps of the policy is 20. 
In this case a heatmap renderer should take into account the relevant metric
for the appropriate display of the policy, learning rate and states history.
'''
def get_extent(env_size):

	CONFIG = [[5, 20], [18, 36]]
	size = 0
	steps = 0

	for c in CONFIG:
		if env_size == c[0]:
			size = c[0]
			steps = c[1]
			break
	
	# extent = [left, right, bottom, top]
	#extent = [-0.5, steps + 0.5, -0.5, size + 0.5]
	extent = [-0.5, steps + 0.5, 0.5, size + 0.5]	
	
	return extent, size, steps

'''
def get_policy_matrix
return - policy in matrix form
'''
def get_policy_matrix(policy, w, h):

	tlen = w + 1

	nested = [0 for i in range(h)]
	data = [nested[:] for i in range(tlen)]

	for i in range(tlen):
		if policy[i] == '' and i == 0:
			continue
		
		index = int(policy[i])
		data[i - 1][index] += 1

	return data


def heatmap(ax, title, hdata, extent, cmap, xlabel, ylabel, vminmax):

	#data = np.arange(10, 0, -(10/648.0)).reshape(36, 18)
	#data = np.zeros(36 * 18).reshape(36, 18)
	data = np.asarray(hdata)
	fdata = np.flip(data.T, 0)

	# create an axes on the right side of ax.
	# The width of cax will be 5% of ax and
	# the padding between cax and ax will be fixed at 0.05 inch.
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size = "2%", pad = 0.075)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
	xticks = [i for i in range(0, 22, 1)]
	yticks = [i for i in range(1, 6, 1)]
	ax.set_xticks(xticks)
	ax.set_yticks(yticks)
	
	ax.set_title(title)
	im = ax.imshow(fdata, cmap = cmap, extent = extent, vmin = vminmax[0], vmax=vminmax[1])
	plt.colorbar(im, cax = cax, orientation = "vertical")
	plt.tight_layout()


def trajectory(ax, policy):

	y_max = [i for i in range(len(policy))]
	y = [i for i in range(len(policy))]

	#
	# rendering
	#

	xticks = ('S1\n|\nO1', 'S2\n|\nO2', 'S3\n|\nO1', 'S4\n|\nO2', 'S5\n|\nO3' )
	yheight = len(policy) + 0.5

	# first figure
	ax.set_title('Max Oneshot')

	plt.plot(policy, y_max, color = 'black', marker = 'o')

	plt.ylabel('Steps')
	plt.ylim(-0.5, yheight)
	plt.xticks(np.arange(5), xticks)
	plt.grid()


'''
def render
-- visualise the policy and its corresponding learning rate and states
in a form of heatmap
'''
def render(name, policy, lr, states, env_size, vminmax):

	xlabel = 'Steps'
	ylabel = 'S-O node index'

	extent, size, steps = get_extent(env_size)
	policy_mat = get_policy_matrix(policy, steps, size)
	
	fig = plt.figure(figsize = (6, 6))
	#fig.suptitle(name)
	
	# state (uncertainty)
	ax = fig.add_subplot(311)
	heatmap(ax, title_s, states, extent, 'jet',  '', ylabel, vminmax[1])
	
	# learning rate
	ax = fig.add_subplot(312)
	heatmap(ax, title_lr, lr, extent, 'jet', '', ylabel, vminmax[0])

	# exploration trajectory
	ax = fig.add_subplot(313)
	heatmap(ax, 'Exploration Policy', policy_mat, extent, 'gray', xlabel, ylabel, vminmax[2])
	
	plt.subplots_adjust(hspace = 0.25)
	plt.tight_layout()
	plt.show()

