import numpy as np
import math
import scipy.stats as stats
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

import lib.os_data_utils_5 as odu

import random_distance as rd
import ast

#
# behaviour data configuration
#

B = 0 # begin
N = 24
T = 20 # max trial
O = 3 # max outcome
S = 5 # number of stimulus

# 
start_trial = 0
end_trial = 20

# total number of rounds per each participant
ROUND = 5

# whether to use Random or not
useRandom = False

# accessed by set_plt_metric()
#plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':12.0, 'figx':3.6, 'figy':4.5}
		
'''
'''
def _to_list_(list_str):

	ret = []
	
	for ls in list_str:
	
		buf = ast.literal_eval(ls)
		ret.append(buf)
		
	return ret
	
#
def run_even_list(nlist, fill_val = np.nan):

    lens = np.array([len(item) for item in nlist])
    mask = lens[:,None] > np.arange(lens.max())

    out = np.full(mask.shape, fill_val)
    out[mask] = np.concatenate(nlist)

    return out
	
	
'''
'''
def compute_rmse(list1, list2):

	arr1 = np.array(list1)
	arr2 = np.array(list2)
	
	rmse = np.sqrt(((arr1 - arr2)**2).mean())
	
	return rmse
	
'''
'''
def compute_os_ic_index(rating_buf):

	os_sub_list = []
	ic_sub_list = []
	
	for item in rating_buf:
	
		# skip if this item is not for 'random' sequence
		if item[0] != 4:
			continue
			
		sub_rating = item[3]
		os_rating = item[4]
		ic_rating = item[5]
		
		rmse_os_sub = compute_rmse(sub_rating, os_rating)
		os_sub_list.append(rmse_os_sub)
		
		rmse_ic_sub = compute_rmse(sub_rating, ic_rating)
		ic_sub_list.append(rmse_ic_sub)
		
	return [os_sub_list, ic_sub_list]
		
'''
for_cls --> True: analyse items in 'cls', False: analyse items outside 'cls'
'''
def result(cls, outlier, efficiency, distance, rd_list = [], rd_indices = [], bOnlyRandom = False):

	ic_os_rd_index_sub = []
	score_sub = []

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
		seqhist.append(shbuf)
		
	# read random sequences and simulation ratings
	df = pd.read_csv('data/os-bayes-random-rating.csv',  converters={'column_name': eval})
	rd_seq = _to_list_(df['0'])

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
		seqh = seqhist[idx]

		# buffer for one subject
		sub_buf = []
		for j in range(start_trial, end_trial):
			trial_buf = []
			
			e = ep[j][0]		# 1, 2, 3, 4 (bayesian, maxos, minos, random)
			seq = seqid[j][0]	# 1-20 (sequence index in the sequence buffer)
			r = rate[j]
			sh = seqh[0][j][0]
			visit = odu.visits_on_each_node(seq - 1, True)
			#print visit
			
			# we skip all procedures if not a random trial
			if (bOnlyRandom == True) and (e != 4):
				continue
			
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
		
			os_rd_rating = []
			bayes_rd_rating = []
			if rd_seq.count(sh.tolist()) > 0:
				location = rd_seq.index(sh.tolist())
				os_rd_rating = ast.literal_eval(df['1'][location])
				bayes_rd_rating = ast.literal_eval(df['2'][location])
				
			trial_buf.append(os_rd_rating)
			trial_buf.append(bayes_rd_rating)
			
			sub_buf.append(trial_buf)
		
		conf_map.extend(sub_buf) # by subject
		
		# oneshot index
		rmse_list = compute_os_ic_index(conf_map)
		ic_os_rd_index_sub.append(rmse_list)
		
		# score
		ic_conf, os_conf, ic_score, os_score = odu.distinct_ic_os_buf(conf_map, efficiency, distance)
		score_sub.append([ic_conf, os_conf, ic_score, os_score])
	
	# all by subjects list
	return ic_os_rd_index_sub, score_sub
	
def render_index(rmse_list):
	
	plt.rcParams["figure.figsize"] = (12.0, 4.0)

	os_sub_mean = []
	os_sub_sem = []
	
	ic_sub_mean = []
	ic_sub_sem = []
	
	ic_os_diff_mean = []
	ic_os_diff_sem = []
	
	for i in range(len(rmse_list)):
	
		os_sub_mean.append(np.mean(rmse_list[i][0]))
		os_sub_sem.append(stats.sem(rmse_list[i][0]))
		
		ic_sub_mean.append(np.mean(rmse_list[i][1]))
		ic_sub_sem.append(stats.sem(rmse_list[i][1]))
		
		ic_os_diff_mean.append(np.mean(rmse_list[i][1]) - np.mean(rmse_list[i][0]))
		
	
	xrange = np.arange(1, 25, 1)
	xticks = xrange.tolist()
	
	plt.errorbar(xrange, os_sub_mean, yerr = os_sub_sem, marker = 'o', linestyle = '--', label = 'Between one-shot agent and subject')
	plt.errorbar(xrange, ic_sub_mean, yerr = ic_sub_sem, marker = 'o', linestyle = '--', label = 'Between Bayes agent and subject')
	#plt.errorbar(xrange, ic_os_diff_mean, marker = 'x', linestyle='dotted', label = 'Distance between the two')
	#plt.plot(xticks, os_sub_mean, marker = 'x', label = 'rmse on ratings: os-subject')
	#plt.plot(xticks, ic_sub_mean, marker = 'x', label = 'rmse on ratings: bayes-subject')
	
	plt.xticks(xrange, xticks)
	plt.xlabel('Subject index')
	plt.ylabel('RMSE between rating patterns', multialignment='center')	
	plt.legend()
	#plt.setp(fontsize = 13)
	
	'''
	xrange = np.arange(1, 25, 1)
	ax_os_sub = plt.subplot(211)
	plt.plot(xrange, os_sub_mean)
	plt.setp(ax_os_sub.get_xticklabels(), visible = False)

	# share x only
	ax_ic_sub = plt.subplot(212, sharex = ax_os_sub)
	plt.plot(xrange, ic_sub_mean)
	# make these tick labels invisible
	plt.setp(ax_ic_sub.get_xticklabels(), fontsize = 13)
	'''
	plt.tight_layout()
	plt.show()
	
'''
cls: a list containing the subject IDs (either os oriented or bayes oriented)
type_os: True when cls is a set of oneshot oriented, False when cls is a set of bayes oriented
outlier:
distance
'''
def plot_opt_vs_counteropt(ic_conf, os_conf, ic_score, os_score, efficiency = False, num_nodes = 5):

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
	#odu.draw_trial_sem(num_nodes, trial_buf_conf, efficiency, pm)
	
	#
	# Score second
	#
	odu.plot_title = 'Score'
	pm = set_plt_metric(efficiency, 'score')
	
	for j in range(len(ic_score)):
		trial_buf_score.append(ic_score[j] + os_score[j])
	
	trial_buf_score = run_even_list(trial_buf_score)
	#odu.draw_trial_sem(num_nodes, trial_buf_score, efficiency, pm)
	
	return trial_buf_score, trial_buf_conf
	

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
				plt_metric['ylim'] = 8.0
				plt_metric['ylow'] = 3.5
				plt_metric['visit_cnt'] = 20
			else:
				plt_metric['ylim'] = 9.1
				plt_metric['ylow'] = 3.0
				plt_metric['visit_cnt'] = 20
			
		else:

			if type == 'confidence':
				plt_metric['ylim'] = 2.6
				plt_metric['ylow'] = 0.0
				plt_metric['visit_cnt'] = 1
			else:
				plt_metric['ylim'] = 3.2
				plt_metric['ylow'] = 0.7
				plt_metric['visit_cnt'] = 1
	
	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
	return plt_metric
	
def render_index_score(index, score, label = '', title = '', xlabel = '', ylabel = ''):

	x = index
	y = score
	
	plt.rcParams["figure.figsize"] = (12.0, 4.0)
	
	fig, ax = plt.subplots()
	
	# title
	plt.title(title)
	
	# label
	plt.xlabel(xlabel, fontsize = 13)
	plt.ylabel(ylabel, fontsize = 13)
	
	# max
	#ylim = max(abs(max(y)), abs(min(y)))
	#plt.ylim(min(y), ylim)
	
	#xlim = max(abs(max(x)), abs(min(x)))
	#plt.xlim(min(x), xlim)
	
	ax.axhline(y = np.median(y), linestyle = '--', linewidth = 0.5, color = 'k')
	ax.axvline(x = np.median(x), linestyle = '--', linewidth = 0.5, color = 'k')
	ax.axvline(x = np.percentile(x, 25), linestyle = '--', linewidth = 0.5, color = 'k')
	ax.axvline(x = np.percentile(x, 75), linestyle = '--', linewidth = 0.5, color = 'k')
	

	#regression part
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	print ('slope = %f, intercept = %f, r_sqaured = %f, p_value = %f\n' % (slope, intercept, r_value**2, p_value))
	line = slope * np.array(x) + intercept
	plt.plot(x, line, 'r', label = 'Linear regression test')#, label='y={:.2f}x+{:.2f}'.format(slope,intercept))
	
	# draw
	plt.scatter(x, y, marker = 's', s = 100, label = 'Square : Subjects')
	
	for i in range(len(x)):
		ax.annotate('%d' % (i + 1), (x[i], y[i]), size = 13)
	
	plt.legend()
	
	plt.tight_layout()
	plt.show()

def render_z_score_rmse_diff(data):

	xrange = np.arange(1, 25, 1)

	plt.errorbar(xrange, data, marker = 'o', linestyle = 'dotted')#, label = 'One-shot index')

	plt.xticks(xrange, xrange.tolist())
	plt.xlabel('Subject index')
	plt.ylabel('One-shot index (Z-Score)')
	plt.legend()
	
	plt.tight_layout()
	plt.show()
	
	
def main():

	os_index_list = []
	os_score_list = []
	os_conf_list = []
	
	bayes_index_list = []
	bayes_score_list = []
	bayes_conf_list = []
	
	ob_index_list = []

	distance = 1
	subj = 0
	#rand_filter, rd_indices = rd.randoms_of_randoms(25)
	
	rmse_list, score_list = result([], [], False, distance, bOnlyRandom = False)
	
	render_index(rmse_list)
	
	for subj in range(len(rmse_list)):
	
		trial_score, trial_conf = plot_opt_vs_counteropt(score_list[subj][0], score_list[subj][1], score_list[subj][2], score_list[subj][3])
		
		os_score_list.append(np.mean(trial_score[1]))
		os_conf_list.append(np.mean(trial_conf[1]))
		os_index_list.append(math.log(1 / np.mean(rmse_list[subj][0])))
		
		bayes_score_list.append(np.mean(trial_score[0]))
		bayes_conf_list.append(np.mean(trial_conf[0]))		
		bayes_index_list.append(math.log(1 / np.mean(rmse_list[subj][1])))
		
		rmse_os = rmse_list[subj][0]
		rmse_b = rmse_list[subj][1]
		#z_bayes = stats.zscore(rmse_list[subj][1])
		tmp_list = []
		for o, b in zip(rmse_os, rmse_b):
			tmp_list.append(b - o)
		ob_index_list.append(np.mean(stats.zscore(tmp_list)))
	
	render_z_score_rmse_diff(ob_index_list)
	
	render_index_score(ob_index_list, stats.zscore(os_conf_list), \
						xlabel = 'Oneshot+ index', ylabel = 'Oneshot+ Causal rating\n(Z-score)')			
	render_index_score(ob_index_list, stats.zscore(bayes_conf_list), \
						xlabel = 'Oneshot+ index', ylabel = 'Bayesian+ Causal rating\n(Z-score)')
	
	render_index_score(ob_index_list, stats.zscore(os_score_list), \
						xlabel = 'Oneshot+ index', ylabel = 'Oneshot+ Test score\n(Z-score)')			
	render_index_score(ob_index_list, stats.zscore(bayes_score_list), \
						xlabel = 'Oneshot+ index', ylabel = 'Bayesian+ Test score\n(Z-score)')
	
	'''
	render_index_score(bayes_index_list, os_conf_list, xlabel = 'Bayesian+ index', ylabel = 'Causal rating - Oneshot+')			
	render_index_score(os_index_list, os_conf_list, xlabel = 'Oneshot+ index', ylabel = 'Causal rating - Oneshot+')
	render_index_score(bayes_index_list, bayes_conf_list, xlabel = 'Bayesian+ index', ylabel = 'Causal rating - Bayesian+')
	render_index_score(os_index_list, bayes_conf_list, xlabel = 'Oneshot+ index', ylabel = 'Causal rating - Bayesian+')
	
	render_index_score(bayes_index_list, os_score_list, xlabel = 'Bayesian+ index', ylabel = 'Test score - Oneshot+')			
	render_index_score(os_index_list, os_score_list, xlabel = 'Oneshot+ index', ylabel = 'Test score - Oneshot+')
	render_index_score(bayes_index_list, bayes_score_list, xlabel = 'Bayesian+ index', ylabel = 'Test score - Bayesian+')
	render_index_score(os_index_list, bayes_score_list, xlabel = 'Oneshot+ index', ylabel = 'Test score- Bayesian+')
	'''
	
	print ('End of Main')
	
if __name__ == '__main__':
	main()


	
