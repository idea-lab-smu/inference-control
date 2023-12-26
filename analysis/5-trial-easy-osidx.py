import numpy as np
import math
import scipy.stats as stats
import scipy.spatial.distance as dist

import lib.os_data_utils_5 as odu
import lib.plot_oneshot_index as pltos

import sequence.random_distance as rd
import matplotlib.pyplot as plt

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

# 
B2R_X2R, X2R_N2R, B2R_N2R = range(3)


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
def get_subject_data(cls, outlier, efficiency, distance, ts = start_trial, te = end_trial):

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
		for j in range(ts, te):
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
	
	return conf_map
	#return odu.distinct_ic_os_buf(conf_map, efficiency, distance)

	
'''
cls: a list containing the subject IDs (either os oriented or bayes oriented)
outlier:
distance
''' 
def plot_novel_vs_nonnovel(cls, cls_type, outlier, efficiency, distance): 

	sub_data = get_subject_data(cls, outlier, efficiency, distance)
	ic_conf, os_conf, ic_score, os_score = odu.distinct_ic_os_buf(sub_data, efficiency, distance)
	
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

	start_trial = 0
	end_trial = 20

	sub_data = get_subject_data(cls, outlier, efficiency, distance, ts = start_trial, te = end_trial)
	ic_conf, os_conf, ic_score, os_score = odu.distinct_ic_os_buf(sub_data, efficiency, distance)

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

	useRandom = False

	if useRandom == True:
		
		plt_metric['std'] = set_plt_metric_random(efficiency)
		
	else:
	
		plt_metric = {'ylim':12.0, 'step':1.0, 'visit_cnt':20, 'std':19.0, 'figx':3.6, 'figy':3.6}
		
		if efficiency == False:
			
			if type == 'confidence':
				plt_metric['ylim'] = 10.0
				plt_metric['ylow'] = 3.5
				plt_metric['visit_cnt'] = 20
			else:
				plt_metric['ylim'] = 10.0
				plt_metric['ylow'] = 3.5
				plt_metric['visit_cnt'] = 20
			
		else:

			if type == 'confidence':
				plt_metric['ylim'] = 3.0
				plt_metric['ylow'] = 0.0
				plt_metric['visit_cnt'] = 1
			else:
				plt_metric['ylim'] = 3.5
				plt_metric['ylow'] = 0.7
				plt_metric['visit_cnt'] = 1
	
	plt_metric['step'] = (plt_metric['ylim'] - plt_metric['ylow']) / float(plt_metric['std']) 
	
	return plt_metric
	

#
# value: the condition we want to check in the data
# index: the index of the data that contains a value
# data: a data that may contain a value at index
#	
def find_item(value, index, data):

	ret = []
	
	for item in data:
		
		if value == item[index]:
			
			ret.append(item)
			
	return ret
	
	
def get_os_index(data, useScore):

	ret = []
	
	for item in data:
	
		if useScore == True:
			confidence = item[3]
		else:
			confidence = item[2]	# 2 for confidence, 3 for score
	
		os = confidence[4]
		if useScore == False:
			ic = np.mean(confidence[0:4])
		else:
			o1 = confidence[0] + confidence[2]
			o2 = confidence[1] + confidence[3]
			ic = np.mean([o1, o2])
		
		osindex = np.exp([os, ic]) / float(sum(np.exp([os, ic])))
		sm_osindex = osindex[0]
		ret.append(osindex[0])
		
	return np.mean(ret)
	
	
# style = os (one-shot) | ic (incremental)
def find_list(os = True, useScore = False):

	osindex_buf = []

	start_trial = 0
	end_trial = 20

	sub_data = get_subject_data([], [], False, 1, \
								ts = start_trial, te = end_trial)

	inc = end_trial - start_trial
	cnt = 0
	
	while cnt < len(sub_data):
	
		subj = sub_data[cnt:cnt + inc]
		subj_osindex = []
		for i in range(1, 5):	# 1: bayes, 2: max-os, 3: min-os, 4: random
			
			buf = find_item(i, 0, subj)
			osindex = get_os_index(buf, useScore)
			subj_osindex.append(osindex)
		
		osindex_buf.append(subj_osindex)
		cnt += inc
		
	ret = []
	for i in range(len(osindex_buf)):
		item = osindex_buf[i]
		if item[1] > 0:
			ret.append(i + 1)

	return ret, osindex_buf, sub_data
	
	
def draw_scatter(x, y, title = '', xlabel = '', ylabel = ''):

	fig, ax = plt.subplots()
	
	# title
	plt.title(title)
	
	# label
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	# max
	ylim = max(abs(max(y)), abs(min(y))) + 1
	plt.ylim(-ylim, ylim)
	
	xlim = max(abs(max(x)), abs(min(x))) + 1
	plt.xlim(-xlim, xlim)
	
	ax.axhline(y = 0, linestyle = '--', linewidth = 0.5, color = 'k')
	ax.axvline(x = 0, linestyle = '--', linewidth = 0.5, color = 'k')
	

	#regression part
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	print ('slope = %f, intercept = %f, r_sqaured = %f, p_value = %f\n' % (slope, intercept, r_value**2, p_value))
	line = slope * np.array(x) + intercept
	plt.plot(x, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
	
	# draw
	plt.scatter(x, y, label = 'index of random sequences')
	
	for i in range(len(x)):
		ax.annotate(' %d' % (i + 1), (x[i], y[i]))
	
	plt.tight_layout()
	plt.show()
		
	
# bm_diff: diff between d_to_bayes and d_to_maxos
def plot_scatter_osindex_over_similarity(os_index, bm_diff, score_idx):

	x = []
	inc = 3
	
	for i in range(len(os_index)):
		s = i * inc
		e = s + inc
		buf = bm_diff[s:e]
		x.append(np.mean(buf))

	y = [oi[score_idx] for oi in os_index]
	
	title = 'Performance over the Sequence Similarity' \
			+ '\n(D(A, B) stands for a distance between A and B)'
	xlabel = 'D(random, bayesian) - D(random, max_os)' \
			+ '\n(positive: closer to max_os)'
	
	if score_idx == 0:
		ylabel = 'OS index for Bayesian Sequences'
	elif score_idx == 1:
		ylabel = 'OS index for Max-OS Sequences'
	elif score_idx == 3:
		ylabel = 'OS index for Random Sequences'
	else:
		ylabel = ''
	
	draw_scatter(x, y, title = title, xlabel = xlabel, ylabel = ylabel)
	
	
# bm_diff: diff between d_to_bayes and d_to_maxos
def plot_scatter_distance_over_similarity(os_index, bm_diff):

	x = []
	inc = 3
	
	for i in range(len(os_index)):
		s = i * inc
		e = s + inc
		buf = bm_diff[s:e]
		x.append(np.mean(buf))

	y_rand_bayes = [abs(i[3] - i[0]) for i in os_index]
	y_rand_maxos = [abs(i[3] - i[1]) for i in os_index]
	
	y = [y_rand_bayes[i] - y_rand_maxos[i] for i in range(len(y_rand_bayes))]
	
	title = 'Performance over the Sequence Similarity' \
			+ '\n(D(A, B) stands for a distance between A and B)'
	xlabel = 'D(random, bayesian) - D(random, max_os)' \
			+ '\n(positive: closer to max_os)'
	ylabel = 'D(rand_os, bayes_os) - D(rand_os, maxos_os)'\
			+ '\n(positive: closer to max_os)'
	
	draw_scatter(x, y, title = title, xlabel = xlabel, ylabel = ylabel)
	
	
def compute_performance(data):

	ret = []
	
	confidence = []
	score = []
	
	for d in data:
	
		confidence.append(d[2])
		score.append(d[3])
		
	conf_T = np.array(confidence).T
	score_T = np.array(score).T
	
	conf_avg = []
	score_avg = []
	
	for i in range(len(conf_T)):
	
		conf_avg.append(np.mean(conf_T[i]))
		score_avg.append(np.mean(score_T[i]))
	
	ret.append(conf_avg)
	ret.append(score_avg)
	
	return ret
	
	
def compute_digest_avg(data):

	ret = []
	
	for d in data:
	
		subj = []
	
		for item in d:
		
			conf_avg = np.mean(item[0])
			
			n1 = item[1][0] + item[1][2]
			n2 = item[1][1] + item[1][3]
			n3 = item[1][4]
			score_avg = np.mean([n1, n2, n3])
			
			subj.append([conf_avg, score_avg])
			
		ret.append(subj)
		
	return ret
	
	
def get_performance_distance(data_avg, mode = B2R_X2R, useScore = False):

	# B2R_X2R, X2R_N2R, B2R_N2R = range(3)
	# 0: bayes, 1: maxos, 2: minos, 3: random
	d_index = [[3, 0, 1], # B2R_X2R
				[3, 1, 2], 
				[3, 0, 2]
			  ]
			  
	mode_idx = d_index[mode]

	d_r2c1 = []
	d_r2c2 = []
	y_diff = []
	
	if useScore == True:
		index = 1
	else:
		index = 0
	
	# 0: bayes, 1: maxos, 2: minos, 3: random
	
	for da in data_avg:
	
		random = mode_idx[0]
		c1 = mode_idx[1]
		c2 = mode_idx[2]
		
		r2c1 = dist.euclidean(da[random][index], da[c1][index])
		r2c2 = dist.euclidean(da[random][index], da[c2][index])
		diff = r2c1 - r2c2
		
		d_r2c1.append(r2c1)
		d_r2c2.append(r2c2)
		y_diff.append(diff)
		
	return d_r2c1, d_r2c2, y_diff
		
	
	
def plot_scatter_performance_over_similarity(tobayes, tomaxos, diff, subj_data, mode = B2R_X2R):

	x_r2b = tobayes
	x_r2os = tomaxos
	sub_data = subj_data
	
	start_trial = 0
	end_trial = 20
	
	digest = []
	
	inc = end_trial - start_trial
	cnt = 0 
	
	while cnt < len(sub_data):
	
		subj = sub_data[cnt: cnt + inc]
		performance = []
		
		for i in range(1, 5):	# 1: bayes, 2: max-os, 3: min-os, 4: random
			
			buf = find_item(i, 0, subj)
			p = compute_performance(buf)
			performance.append(p)
		
		digest.append(performance)
		cnt += inc
		
	digest_avg = compute_digest_avg(digest)
	
	y_r2b, y_r2os, y_diff = get_performance_distance(digest_avg, mode = mode, useScore = True)
	
	
	x = []
	inc = 3
	
	for i in range(len(y_r2b)):
		s = i * inc
		e = s + inc
		buf = diff[s:e]
		x.append(np.mean(buf))
		
	title = 'Performance over the Sequence Similarity' \
			+ '\n(D(A, B) stands for a distance between A and B)'
	xlabel = 'Sequences\nD(random, bayesian) - D(random, max_os)' \
			+ '\n(positive: closer to max_os sequence)'
	ylabel = 'Score\nD(random, bayesian) - D(random, max_os)' \
			+ '\n(positive: closer to max_os)'
	
	draw_scatter(x, y_diff, title = title, xlabel = xlabel, ylabel = ylabel)
	
	'''
	y_r2b, y_r2os, y_diff = get_performance_distance(digest_avg, useScore = False)
	ylabel = 'Confidence\nD(random, bayesian) - D(random, max_os)' \
			+ '\n(positive: closer to max_os)'
	draw_scatter(x, y_diff, title = title, xlabel = xlabel, ylabel = ylabel)
	
	'''
	
	
def get_score(item):

	n1 = item[0] + item[2]
	n2 = item[1] + item[3]
	n3 = item[4]
	score_avg = np.mean([n1, n2, n3])
	
	return score_avg
	
	
def plot_scatter_score_over_diff(subjects, x_r2bayes, x_r2maxos, x_diff):

	X = []
	Y = []
	
	xY = []
	bY = []
	rY = []
	nY = []
	
	r2xX = []
	r2xY = []
	
	num_subj  = len(subjects) / float(20)
	cnt = 0
	
	start_trial = 0
	end_trial = 20
	num_trial = end_trial - start_trial
	
	idiff = 0
	inc = 3
	
	b_seq = rd.seq_bayes_5
	x_seq = rd.seq_maxos_5
	n_seq = rd.seq_minos_5
	
	while cnt < len(subjects):
	
		subject = subjects[cnt:cnt + num_trial:]
		diff = x_diff[idiff * inc : (idiff + 1) * inc]
		diff_r2x = x_r2maxos[idiff * inc : (idiff + 1) * inc]
		r_seq = rd.seq_random_5[idiff * inc : (idiff + 1) * inc]
		
		for i in range(5):
		
			sbuf = subject[i * 4:(i + 1) * 4]
			random = find_item(4, 0, sbuf)[0]	# 1 bayes, 2 maxos, 3 minos, 4 random
			bayes = find_item(1, 0, sbuf)[0]
			maxos = find_item(2, 0, sbuf)[0]
			minos = find_item(3, 0, sbuf)[0]
			
			f_diff_index = random[1] - 10
			r_score = get_score(random[3])
			b_score = get_score(bayes[3])
			x_score = get_score(maxos[3])
			n_score = get_score(minos[3])
			
			r2c1 = dist.euclidean(r_score, b_score)
			r2c2 = dist.euclidean(r_score, x_score)
			
			X.append(diff[f_diff_index])
			r2xX.append(diff_r2x[f_diff_index])
			r2xY.append(r_score - x_score)
			
			Y.append(r2c1 - r2c2)
			xY.append(x_score)
			bY.append(b_score)
			rY.append(r_score)
			nY.append(n_score)

		cnt += num_trial
		idiff += 1

	print ('no error?')
	
	draw_scatter(r2xX, r2xY, xlabel = 'distance from random to maxos', ylabel = 'score diff between random and maxos')
	
	draw_scatter(X, Y, xlabel = 'diff between r2b and r2x', ylabel = 'diff between p_r2b and pr2x')
	draw_scatter(X, xY, xlabel = 'diff between r2b and r2x', ylabel = 'maxos score')
	draw_scatter(X, bY, xlabel = 'diff between r2b and r2x', ylabel = 'bayes score')
	draw_scatter(X, nY, xlabel = 'diff between r2b and r2x', ylabel = 'min score')
	draw_scatter(X, rY, xlabel = 'diff between r2b and r2x', ylabel = 'random score')
	
	
def plot_scatter_all(subjects):

	X = []
	Y = []
	
	xY = []
	bY = []
	rY = []
	nY = []
	
	r2xX = []
	r2xY = []
	
	num_subj  = len(subjects) / float(20)
	cnt = 0
	
	start_trial = 0
	end_trial = 20
	num_trial = end_trial - start_trial
	
	idiff = 0
	inc = 3
	
	b_seq = rd.seq_bayes_5
	x_seq = rd.seq_maxos_5
	n_seq = rd.seq_minos_5
	
	while cnt < len(subjects):
	
		subject = subjects[cnt:cnt + num_trial:]
		diff = x_diff[idiff * inc : (idiff + 1) * inc]
		diff_r2x = x_r2maxos[idiff * inc : (idiff + 1) * inc]
		r_seq = rd.seq_random_5[idiff * inc : (idiff + 1) * inc]
		
		for i in range(5):
		
			sbuf = subject[i * 4:(i + 1) * 4]
			random = find_item(4, 0, sbuf)[0]	# 1 bayes, 2 maxos, 3 minos, 4 random
			bayes = find_item(1, 0, sbuf)[0]
			maxos = find_item(2, 0, sbuf)[0]
			minos = find_item(3, 0, sbuf)[0]
			
			f_diff_index = random[1] - 10
			r_score = get_score(random[3])
			b_score = get_score(bayes[3])
			x_score = get_score(maxos[3])
			n_score = get_score(minos[3])
			
			r2c1 = dist.euclidean(r_score, b_score)
			r2c2 = dist.euclidean(r_score, x_score)
			
			X.append(diff[f_diff_index])
			r2xX.append(diff_r2x[f_diff_index])
			r2xY.append(r_score - x_score)
			
			Y.append(r2c1 - r2c2)
			xY.append(x_score)
			bY.append(b_score)
			rY.append(r_score)
			nY.append(n_score)

		cnt += num_trial
		idiff += 1

	print ('no error?')
	
	draw_scatter(r2xX, r2xY, xlabel = 'distance from random to maxos', ylabel = 'score diff between random and maxos')
	
	draw_scatter(X, Y, xlabel = 'diff between r2b and r2x', ylabel = 'diff between p_r2b and pr2x')
	draw_scatter(X, xY, xlabel = 'diff between r2b and r2x', ylabel = 'maxos score')
	draw_scatter(X, bY, xlabel = 'diff between r2b and r2x', ylabel = 'bayes score')
	draw_scatter(X, nY, xlabel = 'diff between r2b and r2x', ylabel = 'min score')
	draw_scatter(X, rY, xlabel = 'diff between r2b and r2x', ylabel = 'random score')
	
	
def plot_scatter_descriptive(subjects, x_r2bayes, x_r2maxos, x_diff):

	X = []
	r2bY = []
	r2xY = []
	r2nY = []
	
	xticks = []
		
	num_subj  = len(subjects) / float(20)
	cnt = 0
	
	start_trial = 0
	end_trial = 20
	num_trial = end_trial - start_trial
	
	idiff = 0
	inc = 3
	
	while cnt < len(subjects):
	
		subject = subjects[cnt:cnt+num_trial]
		diff = x_diff[idiff * inc : (idiff + 1) * inc]	# can use when plotting by average values
		
		for i in range(5):
		
			sbuf = subject[i * 4:(i + 1) * 4]
			random = find_item(4, 0, sbuf)[0]	# 1 bayes, 2 maxos, 3 minos, 4 random
			bayes = find_item(1, 0, sbuf)[0]
			maxos = find_item(2, 0, sbuf)[0]
			minos = find_item(3, 0, sbuf)[0]
			
			f_diff_index = random[1] - 10
			r_score = get_score(random[3])
			b_score = get_score(bayes[3])
			x_score = get_score(maxos[3])
			n_score = get_score(minos[3])
			
			X.append(idiff + 1)
			r2bY.append(r_score - b_score)
			r2xY.append(r_score - x_score)
			r2nY.append(r_score - n_score)
			
			xticks.append("%d\n%d" % (cnt / num_trial, i))

		cnt += num_trial
		idiff += 1
		
	plt.title('Score Distance between Random and Optimized Sequences')

	legend_txt = ['between random and bayes', 'between random and maxos', 'between random and minos']
	zero = [0 for i in range(120)]
	
	# plot 1
	plt.subplot(3, 1, 1)
	plt.xticks([i for i in range(len(xticks))], xticks, fontsize = 8)
	#plt.xlabel('All trials by all subjects (in sequence)')
	plt.ylabel("Diff on Score")
	plt.plot(r2bY, 'r.--', linewidth = 0.3, label = legend_txt[0])
	plt.plot(zero, 'k')
	plt.legend()
	plt.tight_layout()
	
	# plot 1
	plt.subplot(3, 1, 2)
	plt.xticks([i for i in range(len(xticks))], xticks, fontsize = 8)
	#plt.xlabel('All trials by all subjects (in sequence)')
	plt.ylabel("Diff on Score")
	plt.plot(r2xY, 'g.--', linewidth = 0.3, label = legend_txt[1])
	plt.plot(zero, 'k')
	plt.legend()
	plt.tight_layout()

	# plot 1
	plt.subplot(3, 1, 3)
	plt.xticks([i for i in range(len(xticks))], xticks, fontsize = 8)
	plt.xlabel('All trials by all subjects (in sequence)')
	plt.ylabel("Diff on Score")
	plt.plot(r2nY, 'b.--', linewidth = 0.3, label = legend_txt[2])
	plt.plot(zero, 'k')
	plt.legend()
	plt.tight_layout()

	plt.show()
	
	

def main():

	distance = 1
	
	all = range(1, N + 1)
	os_style, os_index, subj_data = find_list(os = True, useScore = True)
	other_style = [n for n in all if n not in os_style]

	seq_osidx = np.array(os_index).T
	print (odu.paired_t_test(seq_osidx, True))
	
	#plot_scatter_distance_over_similarity(os_index, diff)


	# B2R_X2R, X2R_N2R, B2R_N2R
	tobayes, tomaxos, diff = rd.diff_distance_rand_to_bayes_maxos()
	#plot_scatter_descriptive(subj_data, tobayes, tomaxos, diff)
	#plot_scatter_osindex_over_similarity(os_index, diff, 3)		# random
	plot_scatter_score_over_diff(subj_data, tobayes, tomaxos, diff)	
	plot_scatter_performance_over_similarity(tobayes, tomaxos, diff, subj_data, mode = B2R_X2R)

	tomaxos, tominos, diff = rd.diff_distance_rand_to_maxos_minos()
	plot_scatter_performance_over_similarity(tomaxos, tominos, diff, subj_data, mode = X2R_N2R)

	tobayes, tominos, diff = rd.diff_distance_rand_to_bayes_minos()
	plot_scatter_performance_over_similarity(tobayes, tominos, diff, subj_data, mode = B2R_N2R)

	'''		

	pltos.draw_os_index(os_index, N, useRandom = True)
	pltos.draw_os_index_minmax(os_index, N)
	pltos.draw_os_index_total(os_index, useRandom = True)
	
	# plot the score: opt vs counter-opt sequences
	plot_opt_vs_counteropt(os_style, 5, [], False, distance)		# all data, plot label, outlier, efficiency, confidence, distance
	plot_opt_vs_counteropt(other_style, 5, [], False, distance)		# all data, plot label, outlier, efficiency, confidence, distance
	
	plot_opt_vs_counteropt(os_style, 5, [], True, distance)
	plot_opt_vs_counteropt(other_style, 5, [], True, distance)
	'''


	
if __name__ == '__main__':
	main()


	
