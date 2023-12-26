import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import constant_lr as clr

def render(lr):

	d_hs = []
	
	for item in lr:
		
		d_hs.extend(item)
		
	print d_hs
	plt.hist(d_hs)
	plt.show()
	
		
def render_ext(lrs):

	d_hs = []
	
	for lr in lrs:
	
		for item in lr:
			
			d_hs.extend(item)
		

	range = [0.0, 0.7]
	bins = 50

	h_ns, h_bins, h_patch = plt.hist(d_hs, bins = bins, alpha = 0.6, range = range, label = 'rational')
	
	x_bias = (h_bins[1] - h_bins[0]) / 2.0
	h_x = [i + x_bias for i in h_bins]
	plt.plot(h_x[0:bins], h_ns)

	plt.show()
	
	
def __render_each_lr():

	range = [0.0, 0.7]
	x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0,9, 1.0]
	
	upper_q = np.percentile(lrs[1], 90)
	print upper_q

	for lr in lrs:

		print '---'
		idx = 0
		
		for item in lr:

			if item >= upper_q:
				print item, 'row = ', idx / 5, ' column = ', idx % 5

			idx += 1

	ry, rbins, _ = plt.hist(lrs[0], bins = 10, alpha = 0.6,range = range, density = 1, label = 'rational')
	#y = mlab.normpdf(bins, np.mean(lrs[0]), np.std(lrs[0]))
	#l = plt.plot(bins, y, 'k--', linewidth = 1)
	l = plt.plot(rbins[0:len(rbins)-1], ry, linewidth = 1)
	
	_, pbins, _ = plt.hist(lrs[1],  bins = 10, alpha = 0.6, range = range, density = 1, label = 'psych')
	_, apbins, _ = plt.hist(lrs[2],  bins = 10, alpha = 0.6, range = range, density = 1, label = 'anti-psych')
	
	plt.xlabel("Learning rate")
	plt.ylabel("Frequency")
	plt.legend()
	plt.show()
	
	plt.hist(lrs, bins = 10, alpha = 0.8, label=['rational', 'psych', 'anti-psych'])
	plt.xlabel("Learning rate")
	plt.ylabel("Frequency")
	plt.tight_layout()
	plt.legend()
	plt.show()
	
	
def __render_all_lrs():

	lrs = clr.lrs_rational[:]
	
	render_ext(lrs)
	

if __name__ == "__main__":

	__render_all_lrs()