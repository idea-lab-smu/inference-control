import numpy as np
import math
import scipy.stats as stats

import lib.os_data_utils_17 as odu

#
# behaviour data configuration
#

B = 0 # begin
N = 36
T = 20 # max trial
O = 4 # max outcome
S = 17 # number of stimulus

# 
start_trial = 0
end_trial = 20

# total number of rounds per each participant
ROUND = 5


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
def result(cls, outlier, efficiency, distance):

	episode = []
	rating = []
	seqids = []

	# read data from the files
	for i in range(B, N):
		ebuf, idbuf, abuf = odu.get_mat_data(i)
		episode.append(ebuf)
		seqids.append(idbuf)
		rating.append(abuf)

	#	
	# descriptive statistics
	#	
	
	conf_map = []
	for idx in range(N - B):

		idx_subj = idx + B + 1

		if len(outlier) > 0 and outlier.count(idx_subj) > 0:
			continue

		if len(cls) > 0 and cls.count(idx_subj) <= 0:	
			continue

		ep = episode[idx]
		seqid = seqids[idx]
		rate = rating[idx]

		# buffer for one subject
		sub_buf = []
		for j in range(start_trial, end_trial):
			trial_buf = []
			
			e = ep[j][0]
			seq = seqid[j][0]
			r = rate[j]
			visit = odu.visits_on_each_node(seq - 1, True)
			#print visit
			
			trial_buf.append(e)
			trial_buf.append(seq)
			
			cbuf = [0 for c in range(S)]	# confidence buffer
			sbuf = [0 for c in range(S)]	# score buffer

			for o in range(O):		
				confidence_ep = odu.get_confidence(r[odu.COL[o]], idx)
				odu.update_score(odu.COL[o], visit, cbuf, confidence_ep) 

				score_ep = odu.get_normalised(r[odu.COL[o]], idx)
				odu.update_score(odu.COL[o], visit, sbuf, score_ep)

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
def plot_opt_vs_counteropt(cls, num_nodes, outlier, efficiency, distance):

	ic_conf, os_conf, ic_score, os_score = result(cls, outlier, efficiency, distance)

	trial_buf_conf = [] # idx 0: bayes, idx 1: max-os, idx 2: min-os, idx 3: random
	trial_buf_score = []
	
	offset_bias = 0.0
	
	#
	# confidence first
	#
	odu.plot_title = 'Confidence'
	pm = set_plt_metric(efficiency, 'confidence')

	for i in range(len(ic_conf)):
		trial_buf_conf.append(ic_conf[i] + os_conf[i])
	
	# set the same size of all lists using padding
	trial_buf_conf = run_even_list(trial_buf_conf)
	
	if odu.plot_title == 'Confidence' and efficiency == False:
		offset_bias = 0.05
		
	#odu.draw_trial_sem(num_nodes, trial_buf_conf, efficiency, pm, offset_bias)
	odu.draw_trial_points(num_nodes, trial_buf_conf, efficiency, pm)
	
	#
	# Score second
	#
	odu.plot_title = 'Score'
	pm = set_plt_metric(efficiency, 'score')
	
	for j in range(len(ic_score)):
		trial_buf_score.append(ic_score[j] + os_score[j])
	
	trial_buf_score = run_even_list(trial_buf_score)
	#odu.draw_trial_sem(num_nodes, trial_buf_score, efficiency, pm)
	odu.draw_trial_points(num_nodes, trial_buf_score, efficiency, pm)
	
	#
	# z-score
	#
	#odu.draw_trial_zscore(num_nodes, trial_buf_score, efficiency, pm)
	
	
def set_plt_metric_random(efficiency, type = 'confidence'):

	plt_metric = {'ylow':4.0, 'ylim':8.0, 'step':1.0, 'visit_cnt':36, 'std':19.0, 'figx':4.5, 'figy':3.6}

	if efficiency == False:
	
		plt_metric['ylim'] = 5.1 if type == 'confidence' else 3.0
		plt_metric['ylow'] = 3.5 if type == 'confidence' else 2.3
		plt_metric['visit_cnt'] = 36
		#plt_metric['std'] = 15.0
		
	else:

		plt_metric['ylim'] = 3.0 if type == 'confidence' else 2.0
		plt_metric['ylow'] = 1.5 if type == 'confidence' else 1.2
		plt_metric['visit_cnt'] = 1
		#plt_metric['std'] = 15.0
		
	return plt_metric
	
	
def set_plt_metric(efficiency, type = 'confidence'):

	useRandom = True
	
	if useRandom == True:
	
		plt_metric = set_plt_metric_random(efficiency, type = type)
		
	else:
	
		plt_metric = {'ylow':4.0, 'ylim':8.0, 'step':1.0, 'visit_cnt':36, 'std':19.0, 'figx':3.6, 'figy':3.6}
		
		if efficiency == False:
		
			plt_metric['ylim'] = 4.8 if type == 'confidence' else 2.95
			plt_metric['ylow'] = 3.5 if type == 'confidence' else 2.4
			plt_metric['visit_cnt'] = 36
			#plt_metric['std'] = 15.0
			
		else:


			plt_metric['ylim'] = 4.5 if type == 'confidence' else 1.85
			plt_metric['ylow'] = 2.5 if type == 'confidence' else 1.2
			plt_metric['visit_cnt'] = 1
			#plt_metric['std'] = 15.0

	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std'])
	
	return plt_metric

	
def main():

	distance = 1
	
	# manually clustered
	maxos = [1, 3, 4, 8, 12, 18, 20, 21, 26, 28, 30, 31, 32, 35, 36] # max-os oriented
	bayes = [2, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23, 24, 25, 27, 29, 33, 34]
	# we mean outliers to have the highest score at the random sequence
	outlier = []
	

	# plot the score: opt vs counter-opt sequences
	plot_opt_vs_counteropt([], 17, [], False, distance)		# all data, plot label, outlier, efficiency, confidence, distance
	plot_opt_vs_counteropt([], 17, [], True, distance)
	#plot_novel_vs_nonnovel([], 'Overall', [], False, distance)
	#plot_novel_vs_nonnovel([], 'Overall', [], True, distance)
	'''
	cls = [1, 4, 5, 9, 10, 11, 13, 15, 16, 19, 20, 24]
	plot_opt_vs_counteropt(cls, 17, [], False, distance)		# all data, plot label, outlier, efficiency, confidence, distance
	plot_opt_vs_counteropt(cls, 17, [], True, distance)
	
	cls = [2, 3, 6, 7, 8, 12, 14, 17, 18, 21, 22, 23]
	plot_opt_vs_counteropt(cls, 17, [], False, distance)		# all data, plot label, outlier, efficiency, confidence, distance
	plot_opt_vs_counteropt(cls, 17, [], True, distance)

	plot_opt_vs_counteropt(maxos, 'Max-OS oriented', outlier, False, distance)
	plot_opt_vs_counteropt(maxos, 'Max-OS oriented', outlier, True, distance)
	plot_novel_vs_nonnovel(maxos, 'Max-OS oriented', outlier, False, distance)
	plot_novel_vs_nonnovel(maxos, 'Max-OS oriented', outlier, True, distance)

	plot_opt_vs_counteropt(bayes, 'Bayes oriented', outlier, False, distance)
	plot_opt_vs_counteropt(bayes, 'Bayes oriented', outlier, True, distance)
	plot_novel_vs_nonnovel(bayes, 'Bayes oriented', outlier, False, distance)
	plot_novel_vs_nonnovel(bayes, 'Bayes oriented', outlier, True, distance)
	'''
	
if __name__ == '__main__':
	main()


	
