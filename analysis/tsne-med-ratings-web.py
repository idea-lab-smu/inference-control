import numpy as np
import math
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import itertools
from scipy import linalg
from sklearn.manifold import TSNE
from sklearn import mixture
import matplotlib as mpl

# for PCA
from sklearn.decomposition import PCA

import lib.os_data_utils_5m as odu
import web_medium.medium_analysis as wm
import random_distance as rd

#
# behaviour data configuration
#

B = 0 # begin
N = 28
T = 15 # max trial
O = 3 # max outcome
S = 5 # number of stimulus

# 
start_trial = 0
end_trial = 15

# total number of rounds per each participant
ROUND = 5

# whether to use Random or not
useRandom = False

# supporting sequence history
useSeqHistory = False

# accessed by set_plt_metric()
#plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':12.0, 'figx':3.6, 'figy':4.5}

#
def filtered_by_percentile(conf_map, val_percentile):

	val = 0
	buf = []
	new_conf_map = []
	
	if val_percentile > 0:

		for cm in conf_map:
			buf.extend(cm[2])
		
		confidence_distribution = list(filter(lambda x : x != 0, buf))
		val = np.nanpercentile(confidence_distribution, val_percentile)
		
		for cm in conf_map:
			if max(cm[2]) >= val and max(cm[2]) > min(cm[2]):
				new_conf_map.append(cm)
				
	else:
	
		new_conf_map = conf_map[:]
	
	#print val, ',\t number of excluded trial =, ', 20 - len(new_conf_map)
	
	return new_conf_map
	
#
def run_even_list(nlist, fill_val = np.nan):

    lens = np.array([len(item) for item in nlist])
    mask = lens[:,None] > np.arange(lens.max())

    out = np.full(mask.shape, fill_val)
    out[mask] = np.concatenate(nlist)

    return out

'''
for_cls --> True: analyse items in 'cls', False: analyse items outside 'cls'
'''
def result(cls, outlier, efficiency, distance, rd_list = [], rd_indices = []):

	episode = []
	rating = []
	seqids = []
	seqhist = []

	# read data from the files
	for i in range(B, N):
		ebuf, idbuf, abuf, shbuf = odu.get_mat_data(i)		# shbuf --> sequence history buffer
		episode.append(ebuf)
		seqids.append(idbuf)
		rating.append(abuf)
		#seqhist.append(shbuf)

	#	
	# descriptive statistics
	#	
	
	conf_map = []
	for idx in range(N - B):

		idx_subj = idx + B + 1
		print ('subject id = %d' % idx_subj)

		if len(outlier) > 0 and outlier.count(idx_subj) > 0:
			continue

		if len(cls) > 0 and cls.count(idx_subj) <= 0:	
			continue

		ep = episode[idx]
		seqid = seqids[idx]
		rate = rating[idx]
		#seqh = seqhist[idx]

		# buffer for one subject
		sub_buf = []
		for j in range(start_trial, end_trial):
			trial_buf = []
			
			e = ep[j][0]		# 1, 2, 3, 4 (bayesian, maxos, minos, random)
			seq = seqid[j][0]	# 1-20 (sequence index in the sequence buffer)
			r = rate[j]
			if useSeqHistory == True:
				sh = seqh[0][j][0]
			else:
				sh = np.array([])
			visit = odu.visits_on_each_node(seq - 1, True)
			#print visit
			
			if e == 4:
				rd_idx = rd_list.index(sh.tolist())
				if pure_random.count(rd_idx) == 0:
					continue
			
			trial_buf.append(e)
			trial_buf.append(seq)
	
			cbuf = []

			for o in range(O):		
				print (j, o) # for debugging
				if (o == 2):
					confidence_ep = odu.get_confidence(r[o], idx)
					cbuf.extend(confidence_ep[0:5])
			
			trial_buf.append(cbuf)
			sub_buf.append(trial_buf)
		
		conf_map.append(sub_buf)
	
	return conf_map
	#return odu.distinct_ic_os_buf(conf_map, efficiency, distance)
	
def compute_desc_stat(sub_data):

	s_mean = 0
	s_std = 0
	
	total = []
	
	for sd in sub_data:
	
		if sd[0] == 4:
			continue
			
		score = sd[3]
		total.extend([score[0] + score[2], score[1] + score[3], score[4]])
		
	s_mean = np.mean(total)
	s_std = np.std(total)
	
	return s_mean, s_std
	
def zscore(val, m, std):

	z = (val - m) / float(std)
	
	return z
	
def zscores_over_conditions(conf_map, mu, std):
	
	zscores = [[], [], [], []] 	# bayesian+, oneshot+, oneshot-, random
	
	for cm in conf_map:
	
		#if cm[0] == 4:
		#	continue
	
		idx = cm[0] - 1
		seq_id = cm[1] - 1
		score = cm[3]
		
		zs = [zscore(score[0] + score[2], mu, std), \
			zscore(score[1] + score[3], mu, std), \
			zscore(score[4], mu, std)]
		zscores[idx].extend(zs)

	return zscores
	
def draw_bars(zscores):

	xt = ['Oneshot+', 'Bayesian+', 'Uniform', 'Oneshot-']
	xl = 'Control conditions'
	yl = 'Test score\n(Z score)'
	title = 'Test scores\n(normalized causal rating)'

	plt_metric = {'figx':4.5, 'figy':3.6}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
	#plt.rcParams['font.family']='cursive'

	ind = np.arange(4)
	width = 0.8
	xticks = xt
	ylabel = yl
	
	ylim = [-0.4, 0.24]
	
	#plt.title(title, size = 13)
	plt.xticks(ind, xticks, size = 10)
	#plt.xlabel(xl, size = 13)
	plt.ylabel(ylabel, size = 13)
	plt.ylim(ylim[0], ylim[1])
	
	y = [np.mean(zscores[1]), np.mean(zscores[0]), np.mean(zscores[3]), np.mean(zscores[2])]
	rects = plt.bar(xt, y, width, color = 'gray')
	
	# show!
	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=True,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=True)

	plt.tight_layout()
	plt.show()
	
	
def tsne(cls, num_nodes, outlier, efficiency, distance, rd_list = [], rd_indices = []):

	rating_list = result(cls, outlier, efficiency, distance, rd_list = rd_list)
	web_rating_list = wm.result_cmaps_for_tsne(cls, outlier, efficiency, distance, rd_list = rd_list)
	
	rating_list.extend(web_rating_list)
	
	X_score = []
	X_rating = []
	Y = []
	isubjects = []
	label = ['Bayesian+', 'Oneshot+', 'Oneshot-', 'Uniform']
	
	for rlist in rating_list:
	
		isub = 0
	
		for item in rlist:

			Y.append(item[0] - 1)
			X_rating.append(item[2])
			isubjects.append(isub)
			isub += 1

	# create the object TSNE
	tsne = TSNE(n_components = 2, random_state = 0)
	
	# project the data in 2D
	XR2d = tsne.fit_transform(np.array(X_rating))

	
	# visualize the data
	plt.figure(figsize = (5, 4))
	colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
	
	legend_elements = [Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[0], label='Bayesian+'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[1], label='Oneshot+'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[2], label='Oneshot-'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[3], label='Uniform')]

	gmmbuf = [[], [], [], []]

	for i in range(len(Y)):
		
		#alpha = (isubjects[i] // int(5)) /  4.0
		alpha = 1.0
	
		#if (Y[i] != 3):
		plt.scatter(XR2d[i, 0], XR2d[i, 1], c = colors[Y[i]], alpha = alpha)
		gmmbuf[Y[i]].append(XR2d[i].tolist())
		
	plt.title("Patterns on the causal rating")
	plt.legend(handles=legend_elements)
	plt.tight_layout()
	plt.show()
	
	plt.figure(figsize = (5, 4))
	
	for i in range(len(Y)):
		
		#alpha = (isubjects[i] // int(5)) /  4.0
		alpha = 1.0
	
		if (Y[i] == 1) or (Y[i] == 2):
			plt.scatter(XR2d[i, 0], XR2d[i, 1], c = colors[Y[i]], alpha = alpha)
		
	plt.title("Patterns on the causal rating")
	plt.legend(handles=legend_elements[1:3])
	plt.tight_layout()
	plt.show()
	
	plt.figure(figsize = (5, 4))
	
	for i in range(len(Y)):
		
		#alpha = (isubjects[i] // int(5)) /  4.0
		alpha = 1.0
	
		if (Y[i] < 2):
			plt.scatter(XR2d[i, 0], XR2d[i, 1], c = colors[Y[i]], alpha = alpha)
		
	plt.title("Patterns on the causal rating")
	plt.legend(handles=legend_elements[0:2])
	plt.tight_layout()
	plt.show()
	
	return X_rating, Y, gmmbuf
	
color_iter = itertools.cycle(['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e'])
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

def plot_results(means, covariances, index, title):

    plt.figure(figsize = (5, 4))

    splot = plt.subplot(1, 1, 1 + index)
    label = ['Bayesian+', 'Oneshot+', 'Oneshot-', 'Uniform']
	
    #plt.figure(figsize = (5, 4))

    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
		
        plt.scatter(mean[0], mean[1], color=color)
		
    legend_elements = [Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[0], label='Bayesian+'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[1], label='Oneshot+'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[2], label='Oneshot-'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[3], label='Uniform')]

    plt.xlim(-21, 21)
    plt.ylim(-29, 29)
    #plt.xticks()
    #plt.yticks()
    #plt.title(title)
    plt.legend(handles=legend_elements[0:3])
    plt.tight_layout()
    plt.show()
	
def KLdivergence(x, y):
	"""
	Compute the Kullback-Leibler divergence between two multivariate samples.
	
	Parameters
	----------
	x : 2D array (n,d)
	Samples from distribution P, which typically represents the true
	distribution.
	y : 2D array (m,d)
	Samples from distribution Q, which typically represents the approximate
	distribution.
	
	Returns
	-------
	out : float
	The estimated Kullback-Leibler divergence D(P||Q).
	
	References
	----------
	Pérez-Cruz, F. Kullback-Leibler divergence estimation of
	continuous distributions IEEE International Symposium on Information
	Theory, 2008.
	"""
	from scipy.spatial import cKDTree as KDTree

	# Check the dimensions are consistent
	x = np.atleast_2d(x)
	y = np.atleast_2d(y)

	n,d = x.shape
	m,dy = y.shape

	assert(d == dy)

	# Build a KD tree representation of the samples and find the nearest neighbour
	# of each point in x.
	xtree = KDTree(x)
	ytree = KDTree(y)

	# Get the first two nearest neighbours for x, since the closest one is the
	# sample itself.
	r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
	s = ytree.query(x, k=1, eps=.01, p=2)[0]

	# There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
	# on the first term of the right hand side.
	return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

def bhattacharyya_gaussian_distance(mean1, cov1, mean2, cov2) -> int:
	""" Estimate Bhattacharyya Distance (between Gaussian Distributions)

	Args:
		distribution1: a sample gaussian distribution 1
		distribution2: a sample gaussian distribution 2

	Returns:
		Bhattacharyya distance
	"""
	cov = (1 / 2) * (cov1 + cov2)

	T1 = (1 / 8) * (np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)[0][0])
	T2 = (1 / 2) * np.log(np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))

	return T1 + T2
	
	
def draw_distance(distances):

	xt = ['Oneshot+\nto Oneshot-', 'Oneshot+\nto Bayesian+',]
	xl = 'Control conditions'
	yl = 'Bhattacharyya Distance'
	title = 'Test scores\n(normalized causal rating)'

	plt_metric = {'figx':3.6, 'figy':4.0}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
	#plt.rcParams['font.family']='cursive'

	ind = np.arange(3)
	width = 0.8
	xticks = xt
	ylabel = yl
	
	ylim = [0.0, 0.20]
	
	#plt.title(title, size = 13)
	#plt.xticks(ind, xticks, size = 10)
	#plt.xlabel(xl, size = 13)
	plt.ylabel(ylabel, size = 13)
	plt.ylim(ylim[0], ylim[1])
	
	rects = plt.bar(xt, distances, width, color = 'gray')
	
	# show!
	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=True,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=True)

	plt.tight_layout()
	plt.show()
	
def gmm_modeling(gmmbuf):

	# Fit a Gaussian mixture with EM using two components
	means = []
	covars = []
	for X_train in gmmbuf:
		
		if len(X_train) == 0:
			continue
			
		gmm = mixture.GaussianMixture(n_components = 1, covariance_type='full')
		gmm.fit(X_train)
		p = gmm.predict(X_train)
		
		# plot_results(X_train, p, gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')
		
		print ("MEAN")
		print (gmm.means_)
		print ("VARIANCE")
		print (gmm.covariances_)
		print ('---')

		means.append(gmm.means_.tolist()[0])
		covars.append(gmm.covariances_.tolist()[0])

	plot_results(means[0:3], covars[0:3], 0, 'Gaussian Mixture')
	
	distances = []
	print ("bhattacharyya_gaussian_distance between oneshot+ and oneshot-")
	dist2 = bhattacharyya_gaussian_distance(np.array([means[1]]), np.array(covars[1]), 
									np.array([means[2]]), np.array(covars[2]))
	print (dist2)
	distances.append(dist2)
	
	print ("bhattacharyya_gaussian_distance between Bayesian+ and Oneshot+")
	dist1 = bhattacharyya_gaussian_distance(np.array([means[1]]), np.array(covars[1]), 
									np.array([means[0]]), np.array(covars[0]))
	print (dist1)
	distances.append(dist1)
						
	draw_distance(distances)
	
	
def main_tsne():

	distance = 1
	
	_, _, gmmbuf = tsne([], 5, [], False, distance, rd_list = rd.seq_random_5)
	
	print ("KL divergence:", KLdivergence(gmmbuf[1], gmmbuf[2]))
	
	gmm_modeling(gmmbuf)
	
	'''
	cls = [1, 2, 3, 4, 6, 11, 12, 16, 19, 20, 21, 24] # oneshot index - upper 50 (the larger, the better)
	_ = tsne(cls, 5, [], False, distance, rd_list = rd.seq_random_5)
	
	cls = [5, 7, 8, 9, 10, 13, 14, 15, 17, 18, 22, 23] # oneshot index - lower 50 (rmse)
	_ = tsne(cls, 5, [], False, distance, rd_list = rd.seq_random_5)
	'''
	
def pca(cls, num_nodes, outlier, efficiency, distance, rd_list = [], rd_indices = []):

	rating_list = result(cls, outlier, efficiency, distance, rd_list = rd_list)
	web_rating_list = wm.result_cmaps_for_tsne(cls, outlier, efficiency, distance, rd_list = rd_list)
	
	rating_list.extend(web_rating_list)
	
	X_score = []
	X_rating = []
	Y = []
	isubjects = []
	label = ['Bayesian+', 'Oneshot+', 'Oneshot-', 'Uniform']
	
	for rlist in rating_list:
	
		isub = 0
	
		for item in rlist:

			Y.append(item[0] - 1)
			X_rating.append(item[2])
			isubjects.append(isub)
			isub += 1

	# create the object TSNE
	pca = PCA(n_components = 2)
	
	# project the data in 2D
	pc2d = pca.fit_transform(X_rating)

	# visualize the data
	plt.figure(figsize = (5, 4))
	colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
	
	legend_elements = [Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[0], label='Bayesian+'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[1], label='Oneshot+'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[2], label='Oneshot-'),
					Line2D([0], [0], marker='o', color = 'w', markerfacecolor=colors[3], label='Uniform')]

	gmmbuf = [[], [], [], []]

	for i in range(len(Y)):
		
		#alpha = (isubjects[i] // int(5)) /  4.0
		alpha = 1.0
	
		#if (Y[i] != 3):
		plt.scatter(pc2d[i, 0], pc2d[i, 1], c = colors[Y[i]], alpha = alpha)
		gmmbuf[Y[i]].append(pc2d[i].tolist())
		
	plt.title("Patterns on the causal rating")
	plt.legend(handles=legend_elements)
	plt.tight_layout()
	plt.show()
	
	plt.figure(figsize = (5, 4))
	
	for i in range(len(Y)):
		
		#alpha = (isubjects[i] // int(5)) /  4.0
		alpha = 1.0
	
		if (Y[i] == 1) or (Y[i] == 2):
			plt.scatter(pc2d[i, 0], pc2d[i, 1], c = colors[Y[i]], alpha = alpha)
		
	plt.title("Patterns on the causal rating")
	plt.legend(handles=legend_elements[1:3])
	plt.tight_layout()
	plt.show()
	
	plt.figure(figsize = (5, 4))
	
	for i in range(len(Y)):
		
		#alpha = (isubjects[i] // int(5)) /  4.0
		alpha = 1.0
	
		if (Y[i] < 2):
			plt.scatter(pc2d[i, 0], pc2d[i, 1], c = colors[Y[i]], alpha = alpha)
		
	plt.title("Patterns on the causal rating")
	plt.legend(handles=legend_elements[0:2])
	plt.tight_layout()
	plt.show()
	
	return X_rating, Y, gmmbuf

def main_pca():

	distance = 1

	_, _, gmmbuf = pca([], 5, [], False, distance, rd_list = rd.seq_random_5)
	
	gmm_modeling(gmmbuf)
	
	
if __name__ == '__main__':
	main_tsne()
	#main_pca()

	
