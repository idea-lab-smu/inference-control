import numpy as np
import math
import scipy.stats as stats
import scipy.spatial.distance as dist
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

import lib.os_data_utils_5 as odu
import web_easy.easy_analysis as we

import random_distance as rd
import ast

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
					#print (j, o) # for debugging
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
		#rmse_list = compute_os_ic_index(conf_map)
		rmse_list = compute_os_ic_index(sub_buf)
		ic_os_rd_index_sub.append(rmse_list)
		
		# score
		ic_conf, os_conf, ic_score, os_score = odu.distinct_ic_os_buf(conf_map, efficiency, distance)
		score_sub.append([ic_conf, os_conf, ic_score, os_score])
	
	# all by subjects list
	return ic_os_rd_index_sub, score_sub
	
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
	
	
def main_ext():

	distance = 1
	subj = 0
	
	rmse_list, score_list = result([], [], False, distance, bOnlyRandom = False)
	
	X_train = []
	y_train = []
	
	X_test = []
	y_test = []
	
	for subj in range(len(rmse_list)):
	
		trial_score, trial_conf = plot_opt_vs_counteropt(score_list[subj][0], score_list[subj][1], score_list[subj][2], score_list[subj][3])
		#icmax = trial_score[0]
		#osmax = trial_score[1]
		#osmin = trial_score[2]

		for icon in range(3):
			for i in range(5):
				x = [trial_score[icon][i * 2], trial_score[icon][i * 2 + 1], trial_score[icon][i + 10]]
				#x = [trial_conf[icon][i * 2], trial_conf[icon][i * 2 + 1], trial_conf[icon][i + 10]]
				y = icon
				if i <= 4:
					X_train.append(x)
					y_train.append(y)
				else:
					X_test.append(x)
					y_test.append(y)
					
	logistic = LogisticRegression()
	logistic.fit(X_train, y_train)	
	
	eff_icmax = []
	eff_osmax = []
	eff_osmin = []
	
	test_ic = []
	test_ic_y = []
	test_osmax = []
	test_osmax_y = []
	test_osmin = []
	test_osmin_y = []
	
	for isubj in range(N):

		spos = isubj * 15
		epos = spos + 15
		subj_data = X_train[spos:epos]
		subj_class = y_train[spos:epos]

		for i in range(3):
			cond_x = subj_data[i * 5: (i + 1) * 5] 
			cond_y = subj_class[i * 5: (i + 1) * 5] 
			acc = logistic.score(cond_x, cond_y)
			
			eff_idx = i % 3
			if eff_idx == 0:
				eff_icmax.append(acc)
				test_ic.extend(cond_x)
				test_ic_y.extend(cond_y)
			elif eff_idx == 1:
				eff_osmax.append(acc)
				test_osmax.extend(cond_x)
				test_osmax_y.extend(cond_y)
			else:
				eff_osmin.append(acc)
				test_osmin.extend(cond_x)
				test_osmin_y.extend(cond_y)

				
				
		print('학습용 데이터셋 정확도 : %.2f, %.2f, %.2f' % (eff_icmax[-1], eff_osmax[-1], eff_osmin[-1] ))
	#print('검증용 데이터셋 정확도 : %.2f' % logistic.score(X_test, y_test))
	
	print('mean = ', np.mean(eff_icmax), np.mean(eff_osmax), np.mean(eff_osmin))
	
	from sklearn.metrics import classification_report
	y_pred = logistic.predict(test_ic)
	print(classification_report(test_ic_y, y_pred))
	
	y_pred = logistic.predict(test_osmax)
	print(classification_report(test_osmax_y, y_pred))
	
	y_pred = logistic.predict(test_osmin)
	print(classification_report(test_osmin_y, y_pred))
	
	print('mean = ', np.mean(eff_icmax), np.mean(eff_osmax), np.mean(eff_osmin))
	
	eff_array = np.array(eff_osmax) - np.array(eff_osmin)
	avg_effsize = np.mean(eff_array)
	sem_effsize = stats.sem(eff_array, nan_policy = 'omit')
	print ('average effect size = ', avg_effsize)
	print ('SEM on effect size = ', sem_effsize)
	t_stat, p_val = stats.ttest_ind(eff_array, [0 for i in range(len(eff_osmax))])
	print ('significance on eff: ', p_val)
	
	plt.plot(eff_icmax)
	plt.plot(eff_osmax)
	plt.plot(eff_osmin)
	plt.show()
	
	
	import statsmodels.api as sm
	y = []
	new_x = []
	for yitem, xitem in zip(y_train, X_train):
		if yitem == 0:
			y.append(0)
		else:
			y.append(1)
		new_x.append(xitem)
	#y = [item - 1 if item > 0 else  for item in y_train]
	logit = sm.Logit(y, new_x) #로지스틱 회귀분석 시행
	res = logit.fit()
	print(res.summary2())
	print(np.exp(res.params))
	
	from sklearn import svm
	# Import train_test_split function
	from sklearn.model_selection import train_test_split
	
	new_x = []
	y = []
	
	for yitem, xitem in zip(y_train, X_train):
		if yitem == 0:
			y.append(0)
			new_x.append(xitem)
		elif yitem == 1:
			y.append(1)
			new_x.append(xitem)

	'''
	accs = []
	for i in range(10):
		# Split dataset into training set and test set
		X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=0.3, random_state=i)
	
		clf = svm.SVC(kernel='linear')
		clf.fit(X_train, y_train)
		
		y_pred = clf.predict(X_test)
	
		from sklearn import metrics
	
		# Model Accuracy: how often is the classifier correct?
		acc = metrics.accuracy_score(y_test, y_pred)
		accs.append(acc)
	
	print("mean Accuracy:", np.mean(accs))		
	print ('End of main')

	
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits = 10)
	saccs = []
	for train, test in skf.split(new_x, y):
		clf = svm.SVC(kernel='linear')
		clf.fit(X_train, y_train)
		
		y_pred = clf.predict(X_test)
	
		from sklearn import metrics
	
		# Model Accuracy: how often is the classifier correct?
		acc = metrics.accuracy_score(y_test, y_pred)
		saccs.append(acc)
	
	print("stratified mean Accuracy:", np.mean(saccs))		
	print ('End of main')
	'''
	from sklearn.model_selection import RepeatedStratifiedKFold
	from sklearn.model_selection import cross_val_score

	rskfold = RepeatedStratifiedKFold(n_splits=6, n_repeats=5, random_state=42)
	clf = svm.SVC(kernel='linear')
	scores = cross_val_score(clf, new_x, y, cv=rskfold)
	print (scores.mean())
	
if __name__ == '__main__':
	main_ext()
