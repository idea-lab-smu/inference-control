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
	
	print val, ',\t number of excluded trial =, ', 20 - len(new_conf_map)
	
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

		ep = episode[idx]
		seqid = seqids[idx]
		rate = rating[idx]

		# buffer for one subject
		sub_buf = []
		for j in range(start_trial, end_trial):
			trial_buf = []
			mse = []
			
			e = ep[j][0]
			seq = seqid[j][0]
			r = rate[j]
			
			trial_buf.append(e)
			trial_buf.append(seq)
			
			for o in range(O):		
				confidence_ep = odu.get_confidence(r[odu.COL[o]], idx)
				mse.append(odu.compute_mse(odu.COL[o], confidence_ep))

			trial_buf.append(mse)
			sub_buf.append(trial_buf)
		
		filtered_sub_buf = filtered_by_percentile(sub_buf, 0)
		conf_map.extend(filtered_sub_buf)

	
	# skip at the moment (20180201)
	#draw_os_ic_index(conf_map, False)
	#draw_os_ic_index(conf_map, True)
	
	#return odu.distinct_ic_os_buf(conf_map, efficiency, distance)
	return conf_map

	
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
def plot_mse_opt_vs_counteropt(cls, num_nodes, outlier, efficiency, distance):

	mse_list = result(cls, outlier, efficiency, distance)

	trial_buf_mse = [[], [], [], []] # idx 0: bayes, idx 1: max-os, idx 2: min-os, idx 3: random
		
	# confidence first
	odu.plot_title = 'Mean Squared Error'
	pm = set_plt_metric(efficiency, 'mse')

	for ml in mse_list:
		idx = ml[0] - 1
		trial_buf_mse[idx].extend(ml[2])
		
	odu.draw_trial_sem(num_nodes, trial_buf_mse, efficiency, pm)
	

def set_plt_metric(efficiency, type = 'confidence'):

	plt_metric = {'ylow':4.0, 'ylim':8.0, 'step':1.0, 'visit_cnt':36, 'std':15.5, 'figx':4.5, 'figy':3.6}
	
	if type == 'mse':
		
		plt_metric['ylim'] = 23.5
		plt_metric['ylow'] = 15.0
		plt_metric['visit_cnt'] = 36
		plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
		return plt_metric

	if efficiency == False:
	
		plt_metric['ylim'] = 8.5 if type == 'confidence' else 3.75
		plt_metric['ylow'] = 4.0 if type == 'confidence' else 0.0
		plt_metric['visit_cnt'] = 36
		#plt_metric['std'] = 15.0
		
	else:

		plt_metric['ylim'] = 5.0 if type == 'confidence' else 2.35
		plt_metric['ylow'] = 2.0 if type == 'confidence' else 0.0
		plt_metric['visit_cnt'] = 1
		#plt_metric['std'] = 15.0

	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
	return plt_metric

	
def main():

	# plot the score: opt vs counter-opt sequences
	plot_mse_opt_vs_counteropt([], 17, [], False, 1)		# all data, plot label, outlier, efficiency, confidence, distance

	
if __name__ == '__main__':
	main()


	
