import numpy as np
import math
import scipy.stats as stats

import lib.os_data_utils_5 as odu
import lib.rmse_data as rd
import lib.score_efficiency as eff

import random_distance as drand
import bayes_os_distance as bod


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
	
	print (val, ',\t number of excluded trial =, ', 20 - len(new_conf_map))
	
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
		rand_index_buf = []
		
		for j in range(start_trial, end_trial):
			trial_buf = []
			
			e = ep[j][0]		# 1, 2, 3, 4 (bayesian, maxos, minos, random)
			seq = seqid[j][0]	# 1-20 (sequence index in the sequence buffer)
			r = rate[j]
			sh = seqh[0][j][0]
			visit = odu.visits_on_each_node(seq - 1, True)
			#print visit
			
			trial_buf.append(e)
			trial_buf.append(seq)
			
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
			trial_buf.append(sh.tolist())
			sub_buf.append(trial_buf)
		
		filtered_sub_buf = filtered_by_percentile(sub_buf, 0)
		conf_map.extend(filtered_sub_buf)

	
	# skip at the moment (20180201)
	#draw_os_ic_index(conf_map, False)
	#draw_os_ic_index(conf_map, True)
	
	ic_conf, os_conf, ic_score, os_score =  odu.distinct_ic_os_buf(conf_map, efficiency, distance)
	
	return ic_conf, os_conf, ic_score, os_score, conf_map

	
def rmse_considered(conf_map, target = []):

	ncm = len(conf_map)
	randoms = drand.seq_random_5
	
	cm_rand = []
	
	for cm in conf_map:
		
		buf = []
		
		if cm[0] == 4:	# random
	
			rand_seq = cm[4]
			ri = randoms.index(rand_seq)
			
			if len(target) > 0 and target.count(ri) > 0:
				continue
			
			print (cm[4])
			print (randoms[ri], ri)
			
			buf.append(ri)
			buf.append(cm[2])
			buf.append(cm[3])
			
			cm_rand.append(buf)
		
	return cm_rand
	
def get_performance_metrics(conf_map, efficiency = False):

	Y = [[], []] 	# performance 0: confidence, 1: score

	for cm in conf_map:
		
		c = cm[1]
		s = cm[2]
		
		if efficiency == False:
			denom = 1.0
		else:
			denom = 4.0
		
		cbuf = [c[0]/denom + c[2]/denom, c[1]/denom + c[3], c[4]/denom]
		sbuf = [s[0]/denom + s[2]/denom, s[1]/denom + s[3], s[4]/denom]
		
		Y[0].extend(cbuf)
		Y[1].extend(sbuf)
		
	return Y
			

'''
cls: a list containing the subject IDs (either os oriented or bayes oriented)
type_os: True when cls is a set of oneshot oriented, False when cls is a set of bayes oriented
outlier:
distance
'''
def plot_opt_vs_counteropt(cls, num_nodes, outlier, efficiency, distance, rd_list = [], rd_indices = []):

	ic_conf, os_conf, ic_score, os_score, conf_map = result(cls, outlier, efficiency, distance, rd_list = rd_list, rd_indices = rd_indices)
	
	'''
	u25 = [8, 10, 12, 13, 15, 16, 17, 19, 21, 27, 32, 36, 37, 40, 43, 56, 62, 69]
	u50 = [2, 6, 7, 8, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23, 26, 27, 28, 30, 32, 34, 36, 37, 38, 40, 41, 43, 44, 46, 47, 48, 49, 56, 61, 62, 68, 69]
	[2, 4, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 13, 13, 14, 16, 16, 17, 17, 18, 18, 22, 22, 29, 29, 30, 30, 32, 32, 35, 37, 37, 38, 38, 40, 40, 41, 43, 44, 44, 47, 47, 50, 50, 51, 51, 53, 53, 61, 61, 62, 64, 64, 65, 65, 66, 66, 67, 68, 68, 71, 71]
	u75 = [0, 2, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 56, 57, 58, 61, 62, 64, 66, 67, 68, 69, 70, 71]
	
	cm_rand_u25 = rmse_considered(conf_map, target = u25)
	u25_performance = get_performance_metrics(cm_rand_u25)
	
	cm_rand_u50 = rmse_considered(conf_map, target = u50)
	u50_performance = get_performance_metrics(cm_rand_u50)
	
	cm_rand_u75 = rmse_considered(conf_map, target = u75)
	u75_performance = get_performance_metrics(cm_rand_u75)
	'''
	
	cm_rand = rmse_considered(conf_map)
	full_performance = get_performance_metrics(cm_rand)
	
	rd.rmse_draw_plot_25(cm_rand, False)
	rd.rmse_draw_plot_25(cm_rand, True)
	rd.rmse_draw_plot_ext(cm_rand, False)
	rd.rmse_draw_plot_ext(cm_rand, True)
	#rd.rmse_draw_plot(cm_rand, True)

	'''
	trial_buf_conf = [] # idx 0: bayes, idx 1: max-os, idx 2: min-os, idx 3: random
	trial_buf_score = []
	
	#
	# confidence first
	#
	odu.plot_title = 'Causal rating'
	pm = set_plt_metric(efficiency, 'confidence')

	for i in range(len(ic_conf)):
		trial_buf_conf.append(ic_conf[i] + os_conf[i])
	
	# set the same size of all lists using padding
	trial_buf_conf = run_even_list(trial_buf_conf)
	odu.draw_trial_sem(num_nodes, trial_buf_conf, efficiency, pm)
	
	#
	# Score second
	#
	odu.plot_title = 'Test score'
	pm = set_plt_metric(efficiency, 'score')
	
	for j in range(len(ic_score)):
		trial_buf_score.append(ic_score[j] + os_score[j])
	
	trial_buf_score = run_even_list(trial_buf_score)
	odu.draw_trial_sem(num_nodes, trial_buf_score, efficiency, pm)
	'''
	

def set_plt_metric_random(efficiency):

	plt_metric = {'ylow':4.0, 'ylim':8.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':4.5, 'figy':3.6}

	if efficiency == False:
	
		plt_metric['ylim'] = 10.0 if type == 'confidence' else 10.0
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
				plt_metric['ylim'] = 10.0
				plt_metric['ylow'] = 3.5
				plt_metric['visit_cnt'] = 20
			else:
				plt_metric['ylim'] = 10.0
				plt_metric['ylow'] = 5.0
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
	
	
def plot_all():

	distance = 1
	
	#rand_filter, rd_indices_high, rd_indices_low = bod.find_target_randoms()
	
	plot_opt_vs_counteropt([], 5, [], False, distance)		# all data, plot label, outlier, efficiency, confidence, distance
	#plot_opt_vs_counteropt([], 5, [], True, distance)
	

def main():

	plot_all()
	
if __name__ == '__main__':
	main()


	
