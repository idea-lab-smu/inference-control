import numpy as np
import math
import scipy.stats as stats
import scipy.spatial.distance as dist
import random

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

import random_distance as rd


def plot_scatter(ys, xs, cs):

	bPercentile = True
	bLower = True
	bUpper = True
	
	bCB = False

	max_val = max(cs)
	min_val = min(cs)
	
	pct_val_low = np.percentile(cs, 10)
	pct_val_high = np.percentile(cs, 90)
	
	x = []
	y = []
	colors = []
	cc = []		# color code strings
	
	indices_low = []
	indices_high = []
	indices = []
	
	for i in range(len(cs)):
	
		if bPercentile == True:
		
			if (cs[i] <= pct_val_low) and (bLower == True):

				x.append(xs[i])
				y.append(ys[i])
				indices_low.append(i)
				colors.append(cs[i])
				cc.append('r')
				indices.append(i)
			
			if (cs[i] >= pct_val_high) and (bUpper == True):
		
				x.append(xs[i])
				y.append(ys[i])
				indices_high.append(i)
				colors.append(cs[i])
				cc.append('b')
				indices.append(i)
				
			alpha = 0.9
			
		else:
		
			x.append(xs[i])
			y.append(ys[i])
			#colors.append(str((to_minos[i] / max_val) * 255))
			colors.append(cs[i])
			indices.append(i)
			alpha = 0.9
	
	fig, ax = plt.subplots()
	
	# title
	# plt.title('Similarity of Random Sequences')
	
	# label
	plt.xlabel("Distance to oneshot- sequences\n(The lower, the closer to oneshot-)")
	plt.ylabel("Distance to oneshot+ sequences\n(The lower, the closer to Bayesian+)")
	
	# max
	ylim = max(max(x), max(y))
	y0 = min(min(x), min(y))
	plt.ylim(y0 - 1, ylim + 1)
	plt.xlim(y0 - 1, ylim + 1)
	
	# line
	line = [i for i in range(int(y0 - 10), int(ylim + 10))]
		
	# draw
	plt.plot(line, line, '--', color = 'k', linewidth = 0.5)
	
	if bCB == False:
		plt.scatter(x, y, color = cc,  s=100, label = 'Random sequence index', alpha = alpha)
	else:
		plt.scatter(x, y, c = colors, cmap="jet_r", s=100, label = 'Random sequence index', alpha = alpha)
	
	for i in range(len(y)):
		ax.annotate(' %d' % indices[i], (x[i], y[i]))
	
	if bCB == True:
		cb = plt.colorbar()
		cb.set_label("Distance to Bayesian+ sequences\n (The lower, the closer to Bayesian+)")
	
	plt.tight_layout()
	plt.show()

	return indices_low, indices_high

def find_target_randoms(high = True, low = True):

	seq_bayes = rd.seq_bayes_5[:]
	seq_maxos = rd.seq_maxos_5[:]
	seq_minos = rd.seq_minos_5[:]
	seq_rand = rd.seq_random_5[:]

	#plot_rbos_distance(seq_rand, seq_bayes, seq_maxos, seq_minos, True)
	
	to_bayes, to_maxos, _ = rd.diff_distance_rand_to_bayes_maxos()
	to_bayes, to_minos, _ = rd.diff_distance_rand_to_bayes_minos()
	
	indices_low, indices_high = plot_scatter(to_maxos, to_minos, to_bayes)
	
	tgt_random = []
	
	if high == True:
		tgt_random.extend([seq_rand[i] for i in indices_high])
		
	if low == True:
		tgt_random.extend([seq_rand[i] for i in indices_low])
	
	return tgt_random, indices_high, indices_low
	

if __name__ == '__main__':
	print find_target_randoms(high = True, low = True)