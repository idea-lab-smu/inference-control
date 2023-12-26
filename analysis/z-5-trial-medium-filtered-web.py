import numpy as np
import math
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

import lib.os_data_utils_5m as odu
import web_medium.medium_analysis as wm

import random_distance as rd

#
# behaviour data configuration
#

B = 0 # begin
N = 29
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
			
			trial_buf.append(e)
			trial_buf.append(seq)
			
			if rd_list.count(sh.tolist()) > 0 and e == 4:
				cbuf = [np.nan for c in range(S)]	# confidence buffer
				sbuf = [np.nan for c in range(S)]	# score buffer
			else:	
				cbuf = [0 for c in range(S)]	# confidence buffer
				sbuf = [0 for c in range(S)]	# score buffer

				for o in range(O):		
					print (j, o) # for debugging
					confidence_ep = odu.get_confidence(r[o], idx)
					odu.update_score(o, visit, cbuf, confidence_ep) 

					score_ep = odu.get_normalised(r[o], idx)
					odu.update_score(o, visit, sbuf, score_ep)
			
			trial_buf.append(cbuf)
			trial_buf.append(sbuf)
			sub_buf.append(trial_buf)
		
		conf_map.append(sub_buf)
	
	return conf_map
	#return odu.distinct_ic_os_buf(conf_map, efficiency, distance)
	
def compute_desc_stat(sub_data):

	s_mean = 0
	s_std = 0
	
	total = []
	
	for sd in sub_data:
	
		score = sd[3]
		total.extend([score[0] + score[2], score[1] + score[3], score[4]])
		
	s_mean = np.nanmean(total)
	s_std = np.nanstd(total)
	
	return s_mean, s_std
	
def zscore(val, m, std):

	z = (val - m) / float(std)
	
	return z
	
def zscores_over_conditions(conf_map, mu, std):

	zscores = [[], [], [], []] 	# bayesian+, oneshot+, oneshot-, uniform
	
	for cm in conf_map:
	
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
	
	ylim = [-0.25, 0.20]
	
	#plt.title(title, size = 13)
	plt.xticks(ind, xticks, size = 10)
	#plt.xlabel(xl, size = 13)
	plt.ylabel(ylabel, size = 13)
	plt.ylim(ylim[0], ylim[1])
	
	y = [np.nanmean(zscores[1]), np.nanmean(zscores[0]), np.nanmean(zscores[3]), np.nanmean(zscores[2])]
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
	
	
def z_plot_opt_vs_counteropt(cls, num_nodes, outlier, efficiency, distance, rd_list = [], rd_indices = []):

	rating_list = result(cls, outlier, efficiency, distance)
	web_rating_list = wm.result_cmaps_for_zscore(cls, outlier, efficiency, distance)

	rating_list.extend(web_rating_list)
	
	zs_list = [[], [], [], []] 	# bayesian+, oneshot+, oneshot-, uniform

	for rlist in rating_list:
			
		s_mean, s_std = compute_desc_stat(rlist)
		zscores = zscores_over_conditions(rlist, s_mean, s_std)
		
		for i in range(len(zscores)):
			zs_list[i].extend(zscores[i])
	
	draw_bars(zs_list)
	
	return zs_list
	
	
'''
cls: a list containing the subject IDs (either os oriented or bayes oriented)
type_os: True when cls is a set of oneshot oriented, False when cls is a set of bayes oriented
outlier:
distance
'''
def plot_opt_vs_counteropt(cls, num_nodes, outlier, efficiency, distance, rd_list = [], rd_indices = []):

	ic_conf, os_conf, ic_score, os_score = result(cls, outlier, efficiency, distance, rd_list = rd_list, rd_indices = rd_indices)

	trial_buf_conf = [] # idx 0: bayes, idx 1: max-os, idx 2: min-os, idx 3: random
	trial_buf_score = []
	
	#
	# confidence first
	#
	odu.plot_title = 'Confidence'
	pm = set_plt_metric(efficiency, 'confidence')

	for i in range(len(ic_conf)):
		trial_buf_conf.append(ic_conf[i] + os_conf[i])
	
	# set the same size of all lists using padding
	trial_buf_conf = run_even_list(trial_buf_conf)
	odu.draw_trial_sem(num_nodes, trial_buf_conf, efficiency, pm)
	
	#
	# Score second
	#
	odu.plot_title = 'Score'
	pm = set_plt_metric(efficiency, 'score')
	
	for j in range(len(ic_score)):
		trial_buf_score.append(ic_score[j] + os_score[j])
	
	trial_buf_score = run_even_list(trial_buf_score)
	odu.draw_trial_sem(num_nodes, trial_buf_score, efficiency, pm)
	

def set_plt_metric_random(efficiency):

	plt_metric = {'ylow':4.0, 'ylim':8.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':4.5, 'figy':3.6}

	if efficiency == False:
	
		plt_metric['ylim'] = 11.0 if type == 'confidence' else 11.5
		plt_metric['ylow'] = 2.0 if type == 'confidence' else 3.0
		plt_metric['visit_cnt'] = 20
		#plt_metric['std'] = 15.0
		
	else:

		plt_metric['ylim'] = 3.25 if type == 'confidence' else 4.25
		plt_metric['ylow'] = 0.0 if type == 'confidence' else 0.0
		plt_metric['visit_cnt'] = 1
		
	return plt_metric
	
	
def set_plt_metric(efficiency, type = 'confidence'):

	if useRandom == True:
		
		plt_metric = set_plt_metric_random(efficiency)
		
	else:
	
		plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':3.6, 'figy':3.6}
		
		if efficiency == False:
			
			if type == 'confidence':
				plt_metric['ylim'] = 7.5
				plt_metric['ylow'] = 4.3
				plt_metric['visit_cnt'] = 20
			else:
				plt_metric['ylim'] = 7.75
				plt_metric['ylow'] = 5.0
				plt_metric['visit_cnt'] = 20
			
		else:

			if type == 'confidence':
				plt_metric['ylim'] = 2.6
				plt_metric['ylow'] = 0.0
				plt_metric['visit_cnt'] = 1
			else:
				plt_metric['ylim'] = 2.4
				plt_metric['ylow'] = 1.6
				plt_metric['visit_cnt'] = 1
	
	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
	return plt_metric

	
def main():

	distance = 1
	
	z_plot_opt_vs_counteropt([], 5, [], False, distance)

	# plot the score: opt vs counter-opt sequences
	#plot_opt_vs_counteropt([], 5, [], False, distance, rd_list = rand_filter, rd_indices = rd_indices)		# all data, plot label, outlier, efficiency, confidence, distance
	#plot_opt_vs_counteropt([], 5, [], True, distance, rd_list = rand_filter, rd_indices = rd_indices)
	#plot_novel_vs_nonnovel([], 'Overall', [], False, distance)
	#plot_novel_vs_nonnovel([], 'Overall', [], True, distance)
	
if __name__ == '__main__':
	main()


	
