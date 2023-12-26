import numpy as np
import math
import scipy.stats as stats

import lib.os_data_utils_5m as odu
import lib.render_bars as rb

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
		ebuf, idbuf, abuf, _ = odu.get_mat_data(i)
		episode.append(ebuf)
		seqids.append(idbuf)
		rating.append(abuf)

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
				print (j, o) # for debugging
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
	rb.plot_title = 'Confidence'
	ic_conf = run_even_list(ic_conf)
	os_conf = run_even_list(os_conf)
	rb.show_kb_learning(ic_conf, os_conf, cls_type, efficiency)

	# Score second
	rb.plot_title = 'Score'
	ic_score = run_even_list(ic_score)
	os_score = run_even_list(os_score)
	rb.show_kb_learning(ic_conf, os_conf, cls_type, efficiency)
	
def plot_kb_structure_learning(cls, cls_type, outlier, efficiency, distance): 

	ic_conf, os_conf, ic_score, os_score = result(cls, outlier, efficiency, distance)
	
	# confidence first
	rb.plot_title = 'Confidence'
	ic_conf = run_even_list(ic_conf)
	os_conf = run_even_list(os_conf)
	
	rb.show_kb_learning(ic_conf, os_conf, 'Confidence', efficiency)
	print ('---')
	rb.show_kb_learning_across_condition(ic_conf, os_conf, 'Confidence', efficiency)
	print ('---')

	# Score second
	rb.plot_title = 'Score'
	ic_score = run_even_list(ic_score)
	os_score = run_even_list(os_score)
	
	rb.show_kb_learning(ic_score, os_score, 'Score', efficiency)
	print ('---')
	rb.show_kb_learning_across_condition(ic_score, os_score, 'Score', efficiency)
	print ('---')

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

	useRandom = False

	if useRandom == True:
		
		plt_metric['std'] = set_plt_metric_random(efficiency)
		
	else:
	
		plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':3.6, 'figy':3.6}
		
		if efficiency == False:
			
			if type == 'confidence':
				plt_metric['ylim'] = 7.5
				plt_metric['ylow'] = 4.3
				plt_metric['visit_cnt'] = 20
			else:
				plt_metric['ylim'] = 8.2
				plt_metric['ylow'] = 5.6
				plt_metric['visit_cnt'] = 20
			
		else:

			if type == 'confidence':

				plt_metric['ylim'] = 2.3
				plt_metric['ylow'] = 0.7
				plt_metric['visit_cnt'] = 1
			else:
				plt_metric['ylim'] = 2.7
				plt_metric['ylow'] = 1.4
				plt_metric['visit_cnt'] = 1
	
	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
	return plt_metric

	
def main():

	distance = 1
	
	
	# medium
	# medium
	odu.filename = '../20171219-5-node-all/sbj%d_node5.mat'
	odu.col_rating = 'ans_save_tot'
	odu.col_round = 'trial_info'
	odu.col_trial_detail = 'trial_info_detail'
	odu.col_sequences = ''
	
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

	plot_kb_structure_learning([], 'Overall', [], False, distance)
	plot_kb_structure_learning([], 'Overall', [], True, distance)
	
if __name__ == '__main__':
	main()


	
