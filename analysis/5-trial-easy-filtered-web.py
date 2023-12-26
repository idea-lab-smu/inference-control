import numpy as np
import math
import scipy.stats as stats

import lib.os_data_utils_5 as odu
import web_easy.easy_analysis as we

import random_distance as rd

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
useRandom = True

# supporting sequence history
useSeqHistory = False

# accessed by set_plt_metric()
#plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':12.0, 'figx':3.6, 'figy':4.5}

# pure uniform sampling
pure_random = [2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 22, 29, 30, 32, 35, 37, 38, 40, 41, 43, 44, 47, 50, 51, 53,  61, 62, 64, 65, 66, 67, 68, 71]

# random - lower 50% of RMSE between random and oneshot+ sequence
random_score_rmse_under_50 = [10.0, 10.0, 10.0, 6.666666666666666, 6.25, 9.09090909090909, 2.0, 2.5, 6.666666666666666, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 4.444444444444445, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 10.0, 0.0, 10.0, 10.0, 10.0, 7.5, 10.0, 10.0, 2.5806451612903225, 1.6129032258064515, 2.3376623376623376, 2.631578947368421, 4.642857142857143, 0.0, 10.0, 10.0, 10.0, 8.823529411764707, 10.0, 10.0, 10.0, 5.0, 6.666666666666666, 10.0, 10.0, 10.0, 4.545454545454545, 10.0, 0.0, 2.727272727272727, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 5.294117647058823, 10.0, 5.217391304347826, 4.583333333333333, 0.0, 10.0, 10.0, 9.473684210526315, 10.0, 10.0, 6.666666666666666, 10.0, 10.0, 10.0, 3.8461538461538463, 10.0, 0.0, 2.413793103448276, 4.444444444444445, 0.0, 6.842105263157895, 5.833333333333334, 10.0, 10.0, 5.333333333333333, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.166666666666666, 0.0, 10.0, 6.666666666666666, 0.0, 10.0, 0.5263157894736842, 10.0, 0.0, 5.0, 4.090909090909091, 10.0, 10.0, 10.0, 10.0, 7.777777777777778, 8.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 7.777777777777778, 10.0, 5.555555555555555, 10.0, 4.0, 6.923076923076923, 0.0, 9.047619047619047, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 6.666666666666666, 10.0, 10.0, 10.0, 0.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 0.0, 10.0, 10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 1.946564885496183, 3.333333333333333, 10.0, 0.0, 0.0, 5.333333333333333, 10.0, 0.0]

# * on [1.0, 5.0, 10.0, 3.333333333333333, 3.125, 9.09090909090909, 1.0, 1.25, 6.666666666666666, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 5.0, 5.0, 10.0, 5.5, 6.5, 0.8888888888888888, 6.0, 6.0, 10.0, 7.0, 7.0, 8.0, 6.0, 5.0, 9.0, 6.5, 6.0, 8.0, 0.0, 4.5, 0.0, 7.5, 6.0, 10.0, 3.0, 7.5, 10.0, 1.032258064516129, 0.40322580645161288, 0.7012987012987013, 0.6578947368421052, 3.0178571428571428, 0.0, 10.0, 10.0, 10.0, 8.8235294117647065, 10.0, 10.0, 10.0, 2.5, 6.666666666666666, 5.0, 5.0, 10.0, 1.1363636363636362, 4.0, 0.0, 0.40909090909090906, 2.5, 4.0, 10.0, 9.0, 9.0, 0.0, 2.3823529411764706, 10.0, 3.1304347826086953, 2.520833333333333, 0.0, 7.5, 9.0, 8.526315789473683, 9.0, 9.0, 4.0, 9.0, 9.0, 9.0, 1.9230769230769231, 10.0, 0.0, 0.84482758620689657, 1.7777777777777777, 0.0, 4.4473684210526319, 2.041666666666667, 10.0, 7.0, 2.1333333333333333, 6.0, 6.5, 4.0, 10.0, 9.5, 10.0, 10.0, 5.0416666666666661, 0.0, 10.0, 1.3333333333333333, 0.0, 8.0, 0.026315789473684209, 5.0, 0.0, 2.25, 1.8409090909090911, 9.0, 9.0, 7.0, 5.0, 5.4444444444444446, 8.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 5.444444444444445, 3.5, 1.3888888888888888, 9.0, 1.6000000000000001, 3.1153846153846154, 0.0, 8.5952380952380949, 9.0, 10.0, 10.0, 8.5, 9.0, 10.0, 10.0, 10.0, 3.333333333333333, 5.0, 10.0, 5.0, 0.0, 10.0, 0.0, 2.0, 8.0, 0.0, 3.75, 0.0, 10.0, 5.0, 0.0, 5.0, 5.0, 10.0, 5.0, 5.0, 8.0, 0.0, 1.6545801526717556, 1.3333333333333333, 3.5, 0.0, 0.0, 2.1333333333333333, 4.5, 0.0]
#[5.0, 5.0, 10.0, 3.333333333333333, 6.25, 8.333333333333334, 1.1111111111111112, 1.1111111111111112, 4.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 2.8571428571428568, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 5.0, 0.0, 10.0, 10.0, 10.0, 3.333333333333333, 10.0, 10.0, 1.8181818181818183, 1.0869565217391304, 1.3043478260869565, 1.6666666666666665, 4.642857142857143, 0.0, 10.0, 10.0, 10.0, 8.333333333333334, 10.0, 10.0, 10.0, 2.5, 5.0, 5.0, 5.0, 10.0, 2.2727272727272725, 5.0, 0.0, 1.3636363636363635, 5.0, 10.0, 10.0, 10.0, 10.0, 0.0, 2.6470588235294117, 10.0, 5.454545454545454, 4.4, 0.0, 10.0, 10.0, 9.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 2.7777777777777777, 10.0, 0.0, 1.5909090909090908, 3.076923076923077, 0.0, 6.842105263157895, 2.916666666666667, 10.0, 10.0, 2.6666666666666665, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 10.0, 9.166666666666666, 0.0, 10.0, 3.333333333333333, 0.0, 10.0, 0.5, 10.0, 0.0, 2.5, 2.6470588235294117, 10.0, 10.0, 10.0, 10.0, 8.75, 8.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 6.363636363636363, 5.0, 2.2727272727272725, 10.0, 2.8571428571428568, 3.4615384615384617, 0.0, 9.047619047619047, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.333333333333333, 5.0, 10.0, 5.0, 0.0, 10.0, 0.0, 2.5, 10.0, 0.0, 5.0, 0.0, 10.0, 5.0, 0.0, 5.0, 5.0, 10.0, 5.0, 5.0, 10.0, 0.0, 1.5178571428571428, 2.0, 5.0, 0.0, 0.0, 2.6666666666666665, 5.0, 0.0]

random_score_rmse_under_25 = [10.0, 10.0, 10.0, 6.6666666666666661, 6.25, 9.09090909090909, 2.0, 2.5, 6.666666666666666, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 7.5, 10.0, 10.0, 2.6315789473684208, 4.6428571428571432, 0.0, 8.8235294117647065, 10.0, 10.0, 10.0, 5.0, 6.666666666666666, 10.0, 10.0, 10.0, 3.8461538461538463, 10.0, 0.0, 2.4137931034482758, 4.4444444444444446, 0.0, 6.8421052631578947, 5.8333333333333339, 10.0, 10.0, 5.333333333333333, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.1666666666666661, 0.0, 10.0, 6.6666666666666661, 0.0, 10.0, 5.0, 4.0909090909090908, 10.0, 10.0, 10.0, 10.0, 0.0, 5.0, 10.0, 0.0, 1.946564885496183, 3.333333333333333, 10.0, 0.0, 0.0, 5.333333333333333, 10.0, 0.0, 10.0, 0.0, 10.0]
# * 0n [1.0, 5.0, 10.0, 3.333333333333333, 3.125, 9.09090909090909, 1.0, 1.25, 6.666666666666666, 10.0, 10.0, 10.0, 7.0, 7.0, 8.0, 6.5, 6.0, 8.0, 7.5, 6.0, 10.0, 3.0, 7.5, 10.0, 0.6578947368421052, 3.0178571428571428, 0.0, 8.8235294117647065, 10.0, 10.0, 10.0, 2.5, 6.666666666666666, 5.0, 5.0, 10.0, 1.9230769230769231, 10.0, 0.0, 0.84482758620689657, 1.7777777777777777, 0.0, 4.4473684210526319, 2.041666666666667, 10.0, 7.0, 2.1333333333333333, 6.0, 6.5, 4.0, 10.0, 9.5, 10.0, 10.0, 5.0416666666666661, 0.0, 10.0, 1.3333333333333333, 0.0, 8.0, 2.25, 1.8409090909090911, 9.0, 9.0, 7.0, 5.0, 0.0, 2.0, 8.0, 0.0, 1.6545801526717556, 1.3333333333333333, 3.5, 0.0, 0.0, 2.1333333333333333, 4.5, 0.0, 9.0, 0.0, 9.0]
#[5.0, 5.0, 10.0, 3.333333333333333, 6.25, 8.333333333333334, 1.1111111111111112, 1.1111111111111112, 4.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.333333333333333, 10.0, 10.0, 1.6666666666666665, 4.642857142857143, 0.0, 8.333333333333334, 10.0, 10.0, 10.0, 2.5, 5.0, 5.0, 5.0, 10.0, 2.7777777777777777, 10.0, 0.0, 1.5909090909090908, 3.076923076923077, 0.0, 6.842105263157895, 2.916666666666667, 10.0, 10.0, 2.6666666666666665, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 10.0, 9.166666666666666, 0.0, 10.0, 3.333333333333333, 0.0, 10.0, 2.5, 2.6470588235294117, 10.0, 10.0, 10.0, 10.0, 0.0, 2.5, 10.0, 0.0, 1.5178571428571428, 2.0, 5.0, 0.0, 0.0, 2.6666666666666665, 5.0, 0.0, 10.0, 0.0, 10.0]

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
		seqhist.append(shbuf)

	#	
	# descriptive statistics
	#	
	
	conf_map = []
	for idx in range(N - B):

		idx_subj = idx + B + 1
		#print ('subject id = %d' % idx_subj)

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
		rand_index_buf = []
		
		for j in range(start_trial, end_trial):
			trial_buf = []
			
			e = ep[j][0]		# 1, 2, 3, 4 (bayesian, maxos, minos, random)
			seq = seqid[j][0]	# 1-20 (sequence index in the sequence buffer)
			r = rate[j]
			sh = seqh[0][j][0]
			visit = odu.visits_on_each_node(seq - 1, efficiency)
			#print(visit)
			
			if e == 4:
				rd_idx = rd_list.index(sh.tolist())
				if pure_random.count(rd_idx) == 0:
					#print (rd_idx)
					continue
			
			trial_buf.append(e)
			trial_buf.append(seq)
			
			cbuf = [0 for c in range(S)]	# confidence buffer
			sbuf = [0 for c in range(S)]	# score buffer

			for o in range(O):		
				#print (j, o) # for debugging
				confidence_ep = odu.get_confidence(r[o], idx)
				odu.update_score(o, visit, cbuf, confidence_ep) 

				score_ep = odu.get_normalised(r[o], idx)
				odu.update_score(o, visit, sbuf, score_ep)
			
			trial_buf.append(cbuf)
			trial_buf.append(sbuf)
			trial_buf.append(sh.tolist())
			sub_buf.append(trial_buf)
		
		filtered_sub_buf = filtered_by_percentile(sub_buf, 0)
		conf_map.extend(filtered_sub_buf)

	
	# skip at the moment (20180201)
	#draw_os_ic_index(conf_map, False)
	#draw_os_ic_index(conf_map, True)
	
	return odu.distinct_ic_os_buf(conf_map, efficiency, distance)

def result_org(cls, outlier, efficiency, distance, rd_list = [], rd_indices = []):

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
			if useSeqHistory == True:
				sh = seqh[0][j][0]
			else:
				sh = np.array([])
			visit = odu.visits_on_each_node(seq - 1, True)
			#print (visit)
			
			trial_buf.append(e)
			trial_buf.append(seq)
			
			if rd_list.count(sh.tolist()) > 0 and e == 4:
				cbuf = [np.nan for c in range(S)]	# confidence buffer
				sbuf = [np.nan for c in range(S)]	# score buffer
			else:	
				cbuf = [0 for c in range(S)]	# confidence buffer
				sbuf = [0 for c in range(S)]	# score buffer

				for o in range(O):		
					#print (j, o) # for debugging
					confidence_ep = odu.get_confidence(r[o], idx)
					odu.update_score(o, visit, cbuf, confidence_ep) 

					score_ep = odu.get_normalised(r[o], idx)
					odu.update_score(o, visit, sbuf, score_ep)
			
			trial_buf.append(cbuf)
			trial_buf.append(sbuf)
			sub_buf.append(trial_buf)
		
		filtered_sub_buf = filtered_by_percentile(sub_buf, 0)
		conf_map.extend(filtered_sub_buf)

	
	# skip at the moment (20180201)
	#draw_os_ic_index(conf_map, False)
	#draw_os_ic_index(conf_map, True)
	
	return odu.distinct_ic_os_buf(conf_map, efficiency, distance)

	
'''
cls: a list containing the subject IDs (either os oriented or bayes oriented)
outlier:
distance
''' 
def plot_novel_vs_nonnovel(cls, cls_type, outlier, efficiency, distance): 

	ic_conf, os_conf, ic_score, os_score = result(cls, outlier, efficiency, distance)
	
	# confidence first
	odu.plot_title = 'Confidence'
	ic_conf = run_even_list(ic_conf)
	os_conf = run_even_list(os_conf)
	odu.draw_sem(ic_conf, os_conf, cls_type, efficiency)

	# Score second
	odu.plot_title = 'Score'
	ic_score = run_even_list(ic_score)
	os_score = run_even_list(os_score)
	odu.draw_sem(ic_score, os_score, cls_type, efficiency)


'''
cls: a list containing the subject IDs (either os oriented or bayes oriented)
type_os: True when cls is a set of oneshot oriented, False when cls is a set of bayes oriented
outlier:
distance
'''
def plot_opt_vs_counteropt(cls, num_nodes, outlier, efficiency, distance, \
							rd_list = [], rd_indices = [], use_n_pm = True):

	ic_conf, os_conf, ic_score, os_score = result(cls, outlier, efficiency, distance, rd_list = rd_list, rd_indices = rd_indices)
	wic_conf, wos_conf, wic_score, wos_score = we.get_webeasy_result(cls, outlier, efficiency, distance, rd_list = rd_list, rd_indices = rd_indices)

	for i in range(4):
		ic_conf[i] += wic_conf[i]
		os_conf[i] += wos_conf[i]
		ic_score[i] += wic_score[i]
		os_score[i] += wos_score[i]

	trial_buf_conf = [] # idx 0: bayes, idx 1: max-os, idx 2: min-os, idx 3: random
	trial_buf_score = []
	
	#
	# confidence first
	#
	odu.plot_title = 'Confidence'
	if use_n_pm == True:
		pm = set_plt_metric(efficiency, 'confidence')
	else:
		pm = set_plt_metric_comp(efficiency, 'confidence')

	for i in range(len(ic_conf)):
		trial_buf_conf.append(ic_conf[i] + os_conf[i])
	
	# set the same size of all lists using padding
	trial_buf_conf = run_even_list(trial_buf_conf)
	#odu.draw_trial_sem(num_nodes, trial_buf_conf, efficiency, pm)
	odu.draw_trial_points(num_nodes, trial_buf_conf, efficiency, pm)
		
	#
	# Score second
	#
	odu.plot_title = 'Score'
	if use_n_pm == True:
		pm = set_plt_metric(efficiency, 'score')
	else:
		pm = set_plt_metric_comp(efficiency, 'score')
	
	for j in range(len(ic_score)):
		trial_buf_score.append(ic_score[j] + os_score[j])

	trial_buf_score_random = trial_buf_score[:]
	trial_buf_score = run_even_list(trial_buf_score)
	#odu.draw_trial_sem(num_nodes, trial_buf_score, efficiency, pm)
	odu.draw_trial_points(num_nodes, trial_buf_score, efficiency, pm)
	
	# see the statistical significance between bayes+/oneshot+ and random percentiles
	'''
	print ('---')
	print ('uniform sampling seuquences - pure')
	print ('---')
	if efficiency ==  False:
		trial_buf_score_random[3] = random_score_rmse_under_50
	else:
		eff_buf = [rs / 4.0 for rs in random_score_rmse_under_50]
		trial_buf_score_random[3] = eff_buf
	
	odu.draw_trial_sem(num_nodes, trial_buf_score_random, efficiency, pm)
	odu.draw_trial_points(num_nodes, trial_buf_score_random, efficiency, pm)
	'''
	
	#
	# z-score
	#
	#odu.draw_trial_zscore(num_nodes, trial_buf_score, efficiency, pm)
	
	

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
		
		#plt_metric = set_plt_metric_random(efficiency)
		plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':4.5, 'figy':3.6}
		
	else:
	
		plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':3.6, 'figy':3.6}
		
	if efficiency == False:
		
		if type == 'confidence':
			plt_metric['ylim'] = 7.5 if useRandom == False else 6.8
			plt_metric['ylow'] = 4.3
			plt_metric['visit_cnt'] = 20
		else:
			plt_metric['ylim'] = 8.5 if useRandom == False else 6.5
			plt_metric['ylow'] = 4.3
			plt_metric['visit_cnt'] = 20
		
	else:

		if type == 'confidence':
			plt_metric['ylim'] = 2.6 if useRandom == False else 3.0
			plt_metric['ylow'] = 0.0
			plt_metric['visit_cnt'] = 1
		else:
			plt_metric['ylim'] = 2.6  if useRandom == False else 2.2
			plt_metric['ylow'] = 1.0
			plt_metric['visit_cnt'] = 1
	
	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
	return plt_metric
	
def set_plt_metric_comp(efficiency, type = 'confidence'):

	if useRandom == True:
		
		#plt_metric = set_plt_metric_random(efficiency)
		plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':4.5, 'figy':3.6}
		
	else:
	
		plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':3.6, 'figy':3.6}
		
	if efficiency == False:
		
		if type == 'confidence':
			plt_metric['ylim'] = 7.5 if useRandom == False else 8.8
			plt_metric['ylow'] = 4.3
			plt_metric['visit_cnt'] = 20
		else:
			plt_metric['ylim'] = 10.0 if useRandom == False else 9.5
			plt_metric['ylow'] = 5.0
			plt_metric['visit_cnt'] = 20
		
	else:

		if type == 'confidence':
			plt_metric['ylim'] = 2.6 if useRandom == False else 3.0
			plt_metric['ylow'] = 0.0
			plt_metric['visit_cnt'] = 1
		else:
			plt_metric['ylim'] = 2.6  if useRandom == False else 3.0
			plt_metric['ylow'] = 1.6
			plt_metric['visit_cnt'] = 1
	
	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
	return plt_metric

	
def main():

	distance = 1
	
	#rand_filter, rd_indices = rd.randoms_of_randoms(25)
	
	rand_filter = rd.seq_random_5
	rd_indices = []
	cls = []
	
	print ('normal-cases')
    
    # causal rating / test score
	plot_opt_vs_counteropt([], 5, [], False, distance, rd_list = rand_filter, rd_indices = rd_indices)
    
    # efficiency on both
	plot_opt_vs_counteropt([], 5, [], True, distance, rd_list = rand_filter, rd_indices = rd_indices)
	
	'''
	for high-mid-low effect size
	
	cls = [8, 9, 10, 13, 14, 17, 18, 22]	# for low
	plot_opt_vs_counteropt(cls, 5, [], False, distance, \
						rd_list = rand_filter, rd_indices = rd_indices, use_n_pm = False)
						
	cls = [2, 5, 6, 7, 15, 16, 21, 23]	# for mid
	plot_opt_vs_counteropt(cls, 5, [], False, distance, \
						rd_list = rand_filter, rd_indices = rd_indices, use_n_pm = False)
						
	cls = [1, 3, 4, 11, 12, 19, 20, 24]	# for high
	plot_opt_vs_counteropt(cls, 5, [], False, distance, \
						rd_list = rand_filter, rd_indices = rd_indices, use_n_pm = False)

	
	# plot the score: opt vs counter-opt sequences
	cls = [1, 2, 3, 4, 6, 11, 12, 16, 19, 20, 21, 24] # oneshot index - upper 50 (the larger, the better)
	plot_opt_vs_counteropt(cls, 5, [], False, distance, \
						rd_list = rand_filter, rd_indices = rd_indices, use_n_pm = False)		# all data, plot label, outlier, efficiency, confidence, distance
	plot_opt_vs_counteropt(cls, 5, [], True, distance, \
						rd_list = rand_filter, rd_indices = rd_indices, use_n_pm = False)		# all data, plot label, outlier, efficiency, confidence, distance

	cls = [5, 7, 8, 9, 10, 13, 14, 15, 17, 18, 22, 23] # oneshot index - lower 50 (rmse)
	plot_opt_vs_counteropt(cls, 5, [], False, distance, \
						rd_list = rand_filter, rd_indices = rd_indices, use_n_pm = False)			# all data, plot label, outlier, efficiency, confidence, distance
	plot_opt_vs_counteropt(cls, 5, [], True, distance, \
						rd_list = rand_filter, rd_indices = rd_indices, use_n_pm = False)			# all data, plot label, outlier, efficiency, confidence, distance
	#plot_opt_vs_counteropt([], 5, [], True, distance, rd_list = rand_filter, rd_indices = rd_indices)
	#plot_novel_vs_nonnovel([], 'Overall', [], False, distance)
	#plot_novel_vs_nonnovel([], 'Overall', [], True, distance)
    '''
	
if __name__ == '__main__':
	main()


	
