import numpy as np
import math
import scipy.stats as stats

import lib.os_data_utils_17 as odu
import lib.render_bars as rb

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
	
	#print (val, ',\t number of excluded trial =, ', 20 - len(new_conf_map))
	
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
	
	#return odu.distinct_ic_os_buf(conf_map, efficiency, distance)
	return odu.distinct_buf_by_degree(conf_map, efficiency, distance)
	
	
def plot_kb_structure_learning(cls, cls_type, outlier, efficiency, distance): 

	d3_conf, d2_conf, d1_conf, d3_score, d2_score, d1_score = result(cls, outlier, efficiency, distance)
	
	# confidence first
	odu.plot_title = 'Confidence'
	d3_conf = run_even_list(d3_conf)
	d2_conf = run_even_list(d2_conf)
	d1_conf = run_even_list(d1_conf)
	rb.show_kb_learning_difficult(d3_conf, d2_conf, d1_conf, 'Confidence', efficiency)
	print ('---')
	rb.show_kb_learning_across_condition_difficult(d3_conf, d2_conf, d1_conf, 'Confidence', efficiency)
	print ('---')

	# Score second
	odu.plot_title = 'Score'
	d3_conf = run_even_list(d3_score)
	d2_conf = run_even_list(d2_score)
	d1_conf = run_even_list(d1_score)
	rb.show_kb_learning_difficult(d3_score, d2_score, d1_score, 'Score', efficiency)
	print ('---')
	rb.show_kb_learning_across_condition_difficult(d3_score, d2_score, d1_score, 'Score', efficiency)
	print ('---')

	
def set_plt_metric_random(efficiency, type = 'confidence'):

	plt_metric = {'ylow':4.0, 'ylim':8.0, 'step':1.0, 'visit_cnt':36, 'std':19.0, 'figx':4.5, 'figy':3.6}

	if efficiency == False:
	
		plt_metric['ylim'] = 8.5 if type == 'confidence' else 3.7
		plt_metric['ylow'] = 4.0 if type == 'confidence' else 0.0
		plt_metric['visit_cnt'] = 36
		#plt_metric['std'] = 15.0
		
	else:

		plt_metric['ylim'] = 5.0 if type == 'confidence' else 2.3
		plt_metric['ylow'] = 2.0 if type == 'confidence' else 0.0
		plt_metric['visit_cnt'] = 1
		#plt_metric['std'] = 15.0
		
	return plt_metric
	
	
def set_plt_metric(efficiency, type = 'confidence'):

	useRandom = False
	
	if useRandom == True:
	
		plt_metric = set_plt_metric_random(efficiency, type = type)
		
	else:
	
		plt_metric = {'ylow':4.0, 'ylim':8.0, 'step':1.0, 'visit_cnt':36, 'std':19.0, 'figx':3.6, 'figy':3.6}
		
		if efficiency == False:
		
			plt_metric['ylim'] = 7.0 if type == 'confidence' else 2.625
			plt_metric['ylow'] = 5.8 if type == 'confidence' else 1.5
			plt_metric['visit_cnt'] = 36
			#plt_metric['std'] = 15.0
			
		else:

			plt_metric['ylim'] = 4.5 if type == 'confidence' else 1.65
			plt_metric['ylow'] = 2.5 if type == 'confidence' else 0.8
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
	plot_kb_structure_learning([], 'Overall', [], False, distance)
	plot_kb_structure_learning([], 'Overall', [], True, distance)


if __name__ == '__main__':
	main()


	
