import scipy.io as sci		# in order to open .m files 
import numpy as np			# for the numerical operations
import math
import scipy.stats as stats
import pingouin as pg
import scikit_posthocs as sp

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.markers import TICKDOWN

import lib.score_efficiency as eff


#
# Definition
#
filename = './subject_data/difficult/sbj%d_node17.mat'

col_rating = 'ans_save_tot'
col_round = 'trial_info'
col_trial_detail = 'trial_info_detail'

# column to analyse
COL = [12, 13, 14, 16]

'''
COLS = [[7, 8, 0, 2, 4],
		[9, 10, 1, 3, 5, 7, 9, 10],
		[11, 6],
		[14, 11, 6]]

COLS = [[7, 8],
		[9, 10],
		[11],
		[14]]
'''
COLS = [[7, 8],
		[9, 10],
		[11],
		[14]]
		
SOLS = [[0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0]]
		
# global
plot_title = ''

strconf = 'Causal rating'
strscore = 'Test score\n (normalized causal rating)'
streff = 'Efficiency\n (test score / # of presentations)'


#
# interface
#

def set_plot_title(plt_title):

	plot_title = plt_title

#
# functions
#

def sem(data):

	ret = 0;

	#data_len = len(data)
	#data_len = np.count_nonzero(~np.isnan(data))
	#ret = np.nanstd(data) / math.pow(data_len, 0.5)

	ret = stats.sem(data, nan_policy = 'omit')

	return ret


def get_mat_data(index):

	sbj_file = filename % (index + 1)
	print (sbj_file)
	
	mat_data = sci.loadmat(sbj_file)

	sequence = mat_data[col_round]
	sequence_id = mat_data[col_trial_detail]
	answer = mat_data[col_rating]

	return sequence, sequence_id, answer

def get_dirichlet_alpha(data):

	o1 = np.mean([data[7], data[8]])	#ic
	o2 = np.mean([data[9], data[10]])	#ic
	o3 = data[11] #ic
	o4 = data[14] #os
	o5 = np.mean([data[0], data[1], data[2], data[3], data[4], data[5], \
				data[6], data[12], data[13], data[15], data[16]]) # rest of them (fake)
	
	return o1, o2, o3, o4, o5

	
def dirichlet_mean(data):
	
	o1, o2, o3, o4, o5 = get_dirichlet_alpha(data)

	total = sum([o1, o2, o3, o4, o5])
		
	dm1 = o1 / float(total)
	dm2 = o2 / float(total)
	dm3 = o3 / float(total)
	dm4 = o4 / float(total)
	dm5 = o5 / float(total)
	
	return dm1, dm2, dm3, dm4, dm5

def get_normalised(mat_buf, idx):

	mbuf = mat_buf[0]
	length = len(mbuf)
	ratings = []
	ret = [0 for i in range(length)]

	for j in range(length):
		#if idx < 5:
		#	temp = mbuf[j]
		#else:
		temp = mbuf[j][0]

		if type(temp).__name__ == 'str_':
			if temp.find('f') != -1:
				temp = temp.replace('f', '')
		ratings.append(float(temp))

	total = sum(ratings)
	
	if total > 0:
		dm1, dm2, dm3, dm4, dm5 = dirichlet_mean(ratings)
		o1, o2, o3, o4, o5 = get_dirichlet_alpha(ratings) # dm3, o3 --> os
		
		ret[7] = ret[8] = dm1 * 10.0 / 2.0
		ret[9] = ret[10] = dm2 * 10.0 / 2.0
		ret[11] = dm3 * 10.0 / 2.0
		ret[14] = dm4 * 10.0 / 2.0
		ret[0] = ret[1] = ret[2] = ret[3] = ret[4] = ret[5] = \
		ret[6] = ret[12] = ret[13] = ret[15] = ret[16] = dm5 * 10 / 11.0
		
		# original code
		#ret = [(ratings[i] * 10) / total for i in range(length)]

	'''
	print ('---')
	print (ratings)
	print (ret)
	'''
	
	return ret 

'''
def get_normalised(mat_buf, idx):

	mbuf = mat_buf[0]
	length = len(mbuf)
	ret = []

	for j in range(length):
		#if idx < 5:
		#	temp = mbuf[j]
		#else:
		temp = mbuf[j][0]

		if type(temp).__name__ == 'unicode_':
			if temp.find('f') != -1:
				temp = temp.replace('f', '')
		ret.append(int(temp))

	total = sum(ret)
	
	if total > 0:
		ret = [(ret[i] * 10) / float(total) for i in range(length)]

	return ret
'''


# look at the same point but not normalising - only get a confidence on the selection	
def get_confidence(mat_buf, idx):

	mbuf = mat_buf[0]
	length = len(mbuf)
	ret = []

	for j in range(length):
		#if idx < 5:
		#	temp = mbuf[j]
		#else:
		temp = mbuf[j][0]

		if type(temp).__name__ == 'unicode_':
			if temp.find('f') != -1:
				temp = temp.replace('f', '')
		ret.append(int(temp))

	#total = sum(ret)
	
	#if total > 0:
	#	ret = [(ret[i] * 10) / float(total) for i in range(length)]

	return ret
	
	
def compute_mse(index, conf_buf):

	mse = 0
	arg = COL.index(index)
	
	mse = np.square(np.subtract(conf_buf, SOLS[arg])).mean()
	
	return mse


def update_score_ext(index, visit, conf_buf, norm_elem):

	arg = COL.index(index)

	if index == 12:
		spos = 0
	elif index == 13:
		spos = 2
	elif index == 14:
		spos = 4
	elif index == 16:
		spos = 5

	for c in COLS[arg]:

		if visit[spos] > 0:
			conf_buf[c] = norm_elem[c]
		else:
			conf_buf[c] = 0.0

		spos += 1


def update_score(index, visit, conf_buf, norm_elem):

	arg = COL.index(index)

	for c in COLS[arg]:
		if index == 16 and visit[5] <= 0:
			conf_buf[c] = 0.0
		else:
			conf_buf[c] = norm_elem[c]


def t_test(data):

	pval_buf = []

	# t_stat, p_val = stats.mannwhitneyu(os[0], os[1])
	# p_val = p_val * 2
	t_stat, p_val = stats.ttest_ind(data[0], data[1], equal_var=False)
	print (t_stat, p_val)
	pval_buf.append(p_val)

	t_stat, p_val = stats.ttest_ind(data[0], data[2], equal_var=False)
	print (t_stat, p_val)
	pval_buf.append(p_val)

	t_stat, p_val = stats.ttest_ind(data[1], data[2], equal_var=False)
	print (t_stat, p_val)
	pval_buf.append(p_val)

	print ('\n')
	
	return pval_buf


def paired_t_test(data, useRandom):

	pval_buf = []
	
	# t_stat, p_val = stats.mannwhitneyu(os[0], os[1])
	# p_val = p_val * 2
	t_stat, p_val = stats.ttest_rel(data[0], data[1], nan_policy = 'omit')
	print (t_stat, p_val)
	pval_buf.append(p_val)
	
	t_stat, p_val = stats.ttest_rel(data[1], data[2], nan_policy = 'omit')
	print (t_stat, p_val)
	pval_buf.append(p_val)
	
	t_stat, p_val = stats.ttest_rel(data[0], data[2], nan_policy = 'omit')
	print (t_stat, p_val)
	pval_buf.append(p_val)
	
	if useRandom == True:

		t_stat, p_val = stats.ttest_rel(data[1], data[3], nan_policy = 'omit')
		print (t_stat, p_val)
		pval_buf.append(p_val)
		
		t_stat, p_val = stats.ttest_rel(data[0], data[3], nan_policy = 'omit')
		print (t_stat, p_val)
		pval_buf.append(p_val)

	print ('\n')

	return pval_buf


def stars(p):

	if p < 0.0001:
		res_str = "****"
	elif (p < 0.001):
		res_str = "***"
	elif (p < 0.01):
		res_str = "**" 
	elif (p < 0.05):
		res_str = "*" 
	else:
		res_str = 'n.s.' 

	#return res_str + '\n' + str_parenthesis(str(round(p, 3)))
	#return res_str + '\n' + 'p=' + str(round(p, 3))
	return res_str


def significance_bar(start, end, height, p_value,\
				linewidth = 1.0, markersize = 3, boxpad = 0.3, fontsize = 10, color = 'k'):

	# draw a line with downticks at the ends

	plt.plot([start, end], [height] * 2, '-', color = color,\
			lw = linewidth, marker = TICKDOWN, markeredgewidth = linewidth, markersize = markersize)
	
	# draw the text (stars) with a bounding box that covers up the line
	box = dict(facecolor='1.', edgecolor = 'none', boxstyle = 'Square,pad=' + str(boxpad))
	plt.text(0.5 * (start + end), height, stars(p_value),\
			 ha = 'center', va = 'center', bbox = box, size = fontsize)


def significance(ax, data, pval, rects, rect_idx, ylim = 12.0, step = 1.0, useRandom = False, offset_bias = 0.0):

	if useRandom == True:
		order = [[0, 1], [1, 2], [0, 2], [1, 3], [0, 3]]
	else:
		order = [[1, 0], [1, 2], [0, 2]]

	#offset = plt_metric['step']
	offset = offset_bias + step * 0.85
	height = max_height(data) + (offset * 2.0)

	for pi in range(len(pval)):

		height += step * 1.0
		s_rect = rects[order[pi][0]][rect_idx]
		e_rect = rects[order[pi][1]][rect_idx]

		start = s_rect.get_x() + s_rect.get_width() / 2.0
		end = e_rect.get_x() + e_rect.get_width() / 2.0
		significance_bar(start, end, height, pval[pi])
	

def autolabel(ax, rects, x_label = '', step = 1.0, ylow=0.0, visits = 36):

	if visits > 1:
		str_visit = '%d\nvisits' % visits
	else:
		str_visit = '%d\nvisit' % visits
		
	fontsize = 11

	'''
	Attach a text label above each bar displaying its height
	'''
	for rect in rects:

		height = rect.get_height()
		bbox = rect.get_bbox()

		# exact values
		#ax.text(rect.get_x() + rect.get_width() / 2.0, 1.05 * height, '%.2f' % round(float(height), 3), \
		#		ha='center', va='bottom')

		# xticks
		#ax.text(rect.get_x() + rect.get_width() / 2.0, -4.0, x_label, \
		#		ha='center', va='bottom', rotation=270)
		ax.text(rect.get_x() + rect.get_width() / 2.0, ylow - (step * 0.3), x_label, \
				ha='center', va='top', size=fontsize, color='k')
		
		# visit count on each node		
		#ax.text(rect.get_x() + rect.get_width() / 2.0, ylow + (step * 0.3), str_visit, \
		#		ha='center', va='bottom', size=fontsize, color='w')


def max_height(buf):

	length = len(buf)
	max_buf = []
	for li in range(length):
		max_buf.append(np.nanmean(buf[li]))

	return max(max_buf)


def legend_text(num_item):

	ret = []
	
	non_novel = 'non-novel pair %d'
	novel = 'novel pair %d'

	novel_index = num_item - 1

	for i in range(num_item):

		if i < novel_index:
			ret.append(non_novel % (i + 1))
		else:
			ret.append(novel % (i - novel_index + 1))

	return ret


def draw_index(buf, title):

	x = np.arange(0, ROUND, 1)
	tick_txt = ['Round' + str(i + 1) for i in range(ROUND)]
	legend_txt = legend_text(len(buf)) #['S' + str(i + 1) for i in range(len(buf))]

	fig = plt.figure()

	plt.title(title)
	plt.xlabel('Round')
	plt.xticks(x, tick_txt)
	plt.ylabel('Score')
	plt.ylim(0, 10)

	# we have len(buf) number of S-O pairs in the task
	# 1, 2, 3, 4 --> non-novel pairs, 5 --> novel pair in here.
	for i in range(len(buf)):

		index_mean = []
		index_sem = []
		index_err = []
		

		# each participant takes 5 rounds in the behaviour task
		for j in range(ROUND):	
		
			temp = []

			# we have all n_sub number of participants.
			# In here, we have 9 or 11 subject to the dominant learning strategy of each subject
			n_sub = len(buf[i]) / ROUND
			
			for k in range(n_sub):
				temp.append(buf[i][j + 5 * k])
			index_mean.append(np.nanmean(temp))
			index_sem.append(sem(temp))

		y = np.asarray(index_mean)
		y_err = np.asarray([i / 2.0 for i in index_sem])

		plots = plt.plot(index_mean)
		plt.fill_between(x, y - y_err, y + y_err, alpha = 0.2)

	plt.legend(legend_txt)
	plt.tight_layout()
	plt.show()


def draw_os_ic_index(conf_map, use_adj_score):

	num_questions = 6	# 5-node case: 5, 17-node case: 6

	'''
	actually, we only ask 6 pairs of S-O during 17-nodes case task.
	Thus, checking only for 6 can be valid in this case. 
	''' 

	max_buf = [[] for i in range(num_questions)]
	no_buf = [[] for i in range(num_questions)]
	min_buf = [[] for i in range(num_questions)]
	rand_buf = [[] for i in range(num_questions)]

	n_multiply = 1

	for cm in conf_map:
	
		if (cm[0] - 1) == 0:	# no
			buf = no_buf
		elif (cm[0] - 1) == 1:	# max
			buf = max_buf
		elif (cm[0] - 1) == 2:	# min
			buf = min_buf
		elif (cm[0] - 1) == 3:	# random
			buf = rand_buf
		else:
			return
	
		for i in range(num_questions):
			'''
			we selectively assign the score according to the response design.
			For 5-nodes case, 1 to 5 (actually it is all) will be considered.
			For 17-nodes case, 7, 8, 9, 10, 11 will be treated as a set for non-novel pairs
				and 14 will be treated as a set for a novel pair 
			'''
			if i < 4 and (i % 2 == 0):
				buf[i].append(cm[2][i + 7] + cm[2][i + 8])
			elif i == 4:
				buf[i].append(cm[2][i + 7])
			elif i == 5:
				buf[i].append(cm[2][i + 9])

	draw_index(max_buf, 'Optimal type 1' + '\n' + '(maximising oneshot effect)')
	draw_index(no_buf, 'Optimal type 2' + '\n' + '(bayesian)')
	draw_index(min_buf, 'Counter-optimal type' + '\n' + '(minimising oneshot effect)')
	draw_index(rand_buf, 'Counter-optimal type' + '\n' + '(random)')


def draw_sem(ic, os, title, efficiency):

	no_os_mean = [np.nanmean(ic[0]), np.nanmean(os[0])]
	no_os_err = [sem(ic[0]), sem(os[0])]

	max_os_mean = [np.nanmean(ic[1]), np.nanmean(os[1])]
	max_os_err = [sem(ic[1]), sem(os[1])]

	min_os_mean = [np.nanmean(ic[2]), np.nanmean(os[2])]
	min_os_err = [sem(ic[2]), sem(os[2])]

	rand_os_mean = [np.nanmean(ic[3]), np.nanmean(os[3])]
	rand_os_err = [sem(ic[3]), sem(os[3])]


	# now plotting
	TN = 2
	ind = np.arange(TN)
	#width = 0.275	# setting for 3 rects
	width = 0.2	# settinf for 4 rects
	if efficiency == True:
		y_lim = 11
	else:
		y_lim = 14

	fig, ax = plt.subplots()
	ax.set_ylim(0, y_lim)
	ylabel = plot_title if efficiency == False else '%s-Efficiency' % plot_title
	
	rects1 = ax.bar(ind, no_os_mean, width, yerr = no_os_err, label = 'Bayesian+')
	rects2 = ax.bar(ind + width * 1, max_os_mean, width, yerr = max_os_err, label = 'Oneshot+')
	rects3 = ax.bar(ind + width * 2, min_os_mean, width, yerr = min_os_err, label = 'Oneshot-')
	rects4 = ax.bar(ind + width * 3, rand_os_mean, width, yerr = rand_os_err, label = 'Uniform')

	ax.set_title('%s on S-O pair inference\nOptimised = %s' % (ylabel, title))
	ax.set_ylabel('%s' % ylabel)
	plt.xticks(ind + width * 1.5, ('Non-novel\nPairs', 'Novel\nPair'))
	ax.legend(loc = 'best')

	# annoatating the exact value of each bar
	autolabel(ax, rects1)
	autolabel(ax, rects2)
	autolabel(ax, rects3)
	autolabel(ax, rects4)
	
	ttest_rects = [rects1, rects2, rects3, rects4]

	# statistical significance test & draw significance for IC (idx = 0)
	ic_pval = paired_t_test(ic)
	significance(ax, ic, ic_pval, ttest_rects, 0)

	# statistical significance test & draw significance for OS
	os_pval = paired_t_test(os)
	significance(ax, os, os_pval, ttest_rects, 1)
	
	# show!
	plt.tight_layout()
	plt.show()
	
def draw_trial_zscore(num_nodes, trial_buf, efficiency, plt_metric):

	useRandom = True

	bayes_z = np.nanmean(trial_buf[0])
	max_os_z = np.nanmean(trial_buf[1])
	min_os_z = np.nanmean(trial_buf[2])
	
	if useRandom == True:
		rand_os_z = np.nanmean(trial_buf[3])
		
	z_buf = stats.zscore([max_os_z, bayes_z, rand_os_z, min_os_z])
	
	rects = plt.bar(np.arange(4), z_buf, 0.8, color = 'tab:gray')
	
	xticks = ['Oneshot+', 'Bayesian+', 'Uniform', 'Oneshot-']
	plt.xticks(np.arange(4), xticks)
	plt.ylim(-1.5, 1.5)
	plt.ylabel('Test score' + '\n(Z-score)')
	plt.tight_layout()
	plt.show()

	#draw_trial_sem(num_nodes, [bayes_z, max_os_z, min_os_z, rand_os_z], efficiency, plt_metric, bsem = False)

def draw_trial_sem(num_nodes, trial_buf, efficiency, plt_metric, offset_bias = 0.0):

	useRandom = True

	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])

	bayes_mean = [np.nanmean(trial_buf[0])]
	bayes_err = [sem(trial_buf[0])]

	max_os_mean = [np.nanmean(trial_buf[1])]
	max_os_err = [sem(trial_buf[1])]

	min_os_mean = [np.nanmean(trial_buf[2])]
	min_os_err = [sem(trial_buf[2])]
	
	if useRandom == True:
		rand_os_mean = [np.nanmean(trial_buf[3])]
		rand_os_err = [sem(trial_buf[3])]

	# now plotting
	TN = 1
	ind = np.arange(TN)
	gap = 0.05
	width = 0.2 if useRandom == True else 0.1

	ylim_upper = plt_metric['ylim']
	ylim_bottom = plt_metric['ylow']
	visit_cnt = plt_metric['visit_cnt']

	fig, ax = plt.subplots()
	ax.set_ylim(ylim_bottom, ylim_upper)
	bar_color = 'tab:gray'
	
	if plot_title == 'Confidence':
		ylabel = strconf if efficiency == False else streff
	else:
		ylabel = strscore if efficiency == False else streff
	
	rects1 = ax.bar(ind, max_os_mean, width, yerr = max_os_err, color = bar_color)#, label = 'Bayesian+')
	rects2 = ax.bar(ind + (width + gap) * 1, bayes_mean, width, yerr = bayes_err, color = bar_color)#, label = 'max oneshot')
	rects3 = ax.bar(ind + (width + gap) * 2, min_os_mean, width, yerr = min_os_err, color = bar_color)#, label = 'min oneshot')
	if useRandom == True:
		rects4 = ax.bar(ind + (width + gap) * 3, rand_os_mean, width, yerr = rand_os_err, color = bar_color)#, label = 'random')

	#ax.set_title('Inference task on %d S-O pairs:\n%s' % (num_nodes, ylabel))
	#ax.set_title(title)
	ax.set_ylabel('%s' % ylabel, size = 13)
	ax.set_yticks(np.arange(ylim_bottom, ylim_upper, step = 0.5))
	ax.set_yticklabels(np.arange(ylim_bottom, ylim_upper, step = 0.5), rotation=0, fontsize=13)
	ax.legend(loc = 'best')

	# annoatating the exact value of each bar
	plt.xticks(ind, [])
	autolabel(ax, rects1, 'Oneshot+', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
	autolabel(ax, rects2, 'Bayesian+', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
	autolabel(ax, rects3, 'Oneshot-', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
	if useRandom == True:
		autolabel(ax, rects4, 'Uniform', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])

# oneway ANOVA test
	F_statistic, pVal = stats.f_oneway(trial_buf[1], trial_buf[0], trial_buf[2])
	print ('---')
	print('oneway ANOVA : F={0:.1f}, p={1}'.format(F_statistic, pVal))
	if pVal < 0.05:
		print('Significant!')
	print ('---')
	
	# effect size
	hedges_g = pg.compute_effsize(trial_buf[1], trial_buf[2], eftype = 'hedges')
	print ('effect size (Hedges g) = %f' % hedges_g)
	
	hedges_g = pg.compute_effsize(trial_buf[1], trial_buf[3], eftype = 'hedges')
	print ('effect size (Hedges g) between oneshot+ and uniform = %f' % hedges_g)
	
	hedges_g = pg.compute_effsize(trial_buf[2], trial_buf[3], eftype = 'hedges')
	print ('effect size (Hedges g) between oneshot- and uniform = %f' % hedges_g)
	
	# statistical significance test & draw significance for IC (idx = 0)
	print ('paired t-test')
	trial_pval = paired_t_test([trial_buf[1], trial_buf[0], trial_buf[2], trial_buf[3]], useRandom)
	print ('---')
	
	if useRandom == True:
		ttest_rects = [rects1, rects2, rects3, rects4]
	else:
		ttest_rects = [rects1, rects2, rects3]
	significance(ax, [trial_buf[1], trial_buf[0], trial_buf[2], trial_buf[3]],\
				trial_pval, ttest_rects , 0, ylim=ylim_upper, step=plt_metric['step'], useRandom=useRandom, offset_bias = offset_bias)

	# show!
	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	plt.tight_layout()
	plt.show()
	
# seaborn based plots with individuals

def draw_trial_points(num_nodes, trial_buf, efficiency, plt_metric, bsem = True):

	useRandom = True

	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
		
	# now plotting
	TN = 1
	ind = np.arange(TN)
	gap = 0.05
	if useRandom == True:
		width = 0.2	# setting for 4 rects
	else:
		width = 0.1		# settinf for 3 rects

	if efficiency == True:
		ylim_upper = 13.0
		ylim_bottom = -1.0
	else:
		ylim_upper = 15.0
		ylim_bottom = -1.0
	visit_cnt = plt_metric['visit_cnt']

	fig, ax = plt.subplots()
	ax.set_ylim(ylim_bottom, ylim_upper)
	bar_color = 'tab:gray'
	
	if plot_title == 'Confidence':
		ylabel = strconf if efficiency == False else streff
	else:
		ylabel = strscore if efficiency == False else streff
        
	# individual data points
	fulldata = [trial_buf[1], trial_buf[0], trial_buf[3], trial_buf[2]]
	#sns.violinplot(data = fulldata)
	#sns.barplot(data = fulldata, capsize = .1, color = bar_color, ci = 68)
	sns.boxplot(data = fulldata, color = '1')
	#sns.swarmplot(data = fulldata, color = '0', alpha= 0.35)
	sns.stripplot(data = fulldata, jitter = 0.2, edgecolor='gray', color = '0', alpha= 0.1)#facecolors='none', )

	ax.set_ylabel('%s' % ylabel, size = 13)
	ax.set_yticks(np.arange(ylim_bottom, ylim_upper, step = 1.0))
	ax.set_yticklabels(np.arange(ylim_bottom, ylim_upper, step = 1.0), rotation=0, fontsize=12)
	ax.legend(loc = 'best')
    
	tick_txt = ['Oneshot+', 'Bayesian+', 'Uniform', 'Oneshot-']
	x = np.arange(0, len(tick_txt), 1)
	#ax.set_xticks(tick_txt)
	ax.set_xticklabels(tick_txt)
	
	st_res = stats.kruskal(trial_buf[1], trial_buf[0], trial_buf[2])
	print ('Kruskal Wallis (non-param) ---')
	print (st_res)
	
	p_values= sp.posthoc_dunn(fulldata, p_adjust = 'holm')
	print ('Posthoc analysis of Kruskal Wallis ---')
	print (p_values)
	
	print("OS+ vs Bayes+")
	print(stats.mannwhitneyu(trial_buf[0], trial_buf[1]))
	print("OS+ vs Random")
	print(stats.mannwhitneyu(trial_buf[1], trial_buf[3], nan_policy = 'omit'))
	print("OS+ vs OS-")
	print(stats.mannwhitneyu(trial_buf[1], trial_buf[2]))
	print("Bayes+ vs OS-")
	print(stats.mannwhitneyu(trial_buf[0], trial_buf[2]))
	print("Bayes+ vs Random")
	print(stats.mannwhitneyu(trial_buf[0], trial_buf[3], nan_policy = 'omit'))
	print("OS- vs Random")
	print(stats.mannwhitneyu(trial_buf[2], trial_buf[3], nan_policy = 'omit'))

	
	'''
	# annoatating the exact value of each bar
	autolabel(ax, rects1, 'Oneshot+', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
	autolabel(ax, rects2, 'Bayesian+', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
	
	if useRandom == True:
		autolabel(ax, rects3, 'Uniform', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
		autolabel(ax, rects4, 'Oneshot-', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
	else:
		autolabel(ax, rects3, 'Oneshot-', step=plt_metric['step'], ylow=ylim_bottom, visits = plt_metric['visit_cnt'])
		
	
	# oneway ANOVA test
	F_statistic, pVal = stats.f_oneway(trial_buf[1], trial_buf[0], trial_buf[2], trial_buf[3])
	print ('---')
	print('oneway ANOVA : F={0:.1f}, p={1}'.format(F_statistic, pVal))
	if pVal < 0.05:
		print('Significant!')
	print ('---')
	
	# effect size
	hedges_g = pg.compute_effsize(trial_buf[1], trial_buf[2], eftype = 'hedges')
	print ('effect size between os+ and os- (Hedges g) = %f' % hedges_g)
	print ('---')
	
	# statistical significance test & draw significance for IC (idx = 0)
	print ('paired t-test')
	if useRandom == True:
		trial_pval = paired_t_test([trial_buf[1], trial_buf[0], trial_buf[3], trial_buf[2]], useRandom)	
		hedges_g = pg.compute_effsize(trial_buf[1], trial_buf[3], eftype = 'hedges')
		print ('effect size between os+ and uniform (Hedges g) = %f' % hedges_g)
	else:
		trial_pval = paired_t_test([trial_buf[1], trial_buf[0], trial_buf[2]], useRandom)
	print ('---')
	
    '''
					
	# show!
	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)
	plt.tight_layout()
	plt.show()

	


def __validate__(score, visit_cnt):

	res = 0.0

	if visit_cnt > 0.0:

		res = score / visit_cnt

	return res	
	

def distinct_ic_os_buf(conf_map, efficiency, distance):

	# we see the result
	os_conf = [[], [], [], []]	# no, max, min, random
	ic_conf = [[], [], [], []]	# no, max, min, random
	
	# we see the result
	os_score = [[], [], [], []]	# no, max, min, random
	ic_score = [[], [], [], []]	# no, max, min, random
	
	if plot_title == 'Confidence':
		mul = 1.0
	else:
		mul = 1.0

	for cm in conf_map:

		# representing which category this sequence is in 
		# (e.g. bayesian, maxos, minos, random)
		seq_index = cm[0] - 1	

		# representing which sequence is used for this
		seq_id = cm[1] - 1		

		# 7->12, 8->12, 9->13, 10->13, 11->14, 14->16
		visit_cnt = eff.visits_on_node_17(seq_id, efficiency)
		
		for k in range(17):

			if distance == 2:

				if k == 11:
					os_conf[seq_index].append(cm[2][k])
					os_score[seq_index].append(cm[3][k])
				elif k == 6:
					ic_conf[seq_index].append(cm[2][k])
					ic_score[seq_index].append(cm[3][k])
				elif k == 5 or k == 4:
					ic_conf[seq_index].append(cm[2][k] + cm[2][k - 2] + cm[2][k - 4])
					ic_score[seq_index].append(cm[3][k] + cm[3][k - 2] + cm[3][k - 4])

			else:
			
				if k == 14:
					os_conf[seq_index].append(__validate__(cm[2][k], visit_cnt[5]))
					os_score[seq_index].append(__validate__(cm[3][k], visit_cnt[5]))
				elif k == 11:
					ic_conf[seq_index].append(__validate__(cm[2][k], visit_cnt[4]))
					ic_score[seq_index].append(__validate__(cm[3][k], visit_cnt[4]))
				elif k == 10: 
					ic_conf[seq_index].append(__validate__(cm[2][k], visit_cnt[3]) / 2.0 * mul \
										+ __validate__(cm[2][k - 1], visit_cnt[2]) / 2.0 * mul)
					ic_score[seq_index].append(__validate__(cm[3][k], visit_cnt[3]) \
										+ __validate__(cm[3][k - 1], visit_cnt[2]))
				elif k == 8: 
					ic_conf[seq_index].append(__validate__(cm[2][k], visit_cnt[1]) / 2.0 * mul \
										+ __validate__(cm[2][k - 1], visit_cnt[0]) / 2.0 * mul)
					ic_score[seq_index].append(__validate__(cm[3][k], visit_cnt[1]) \
										+ __validate__(cm[3][k - 1], visit_cnt[0]))

	return ic_conf, os_conf, ic_score, os_score
	
def distinct_buf_by_degree(conf_map, efficiency, distance):

	# 12, 13 --> type 3 [4, 6)
	# 14 --> type 2 [2, 4)
	# 16 --> type 1 [0, 2)

	# we see the result
	d3_conf = [[], [], [], []]	# no, max, min, random
	d2_conf = [[], [], [], []]	# no, max, min, random
	d1_conf = [[], [], [], []]	# no, max, min, random
	
	# we see the result
	d3_score = [[], [], [], []]	# no, max, min, random
	d2_score = [[], [], [], []]	# no, max, min, random
	d1_score = [[], [], [], []]	# no, max, min, random
	
	mul = 1.0

	for cm in conf_map:

		# representing which category this sequence is in 
		# (e.g. bayesian, maxos, minos, random)
		seq_index = cm[0] - 1	

		# representing which sequence is used for this
		seq_id = cm[1] - 1		

		# 7->12, 8->12, 9->13, 10->13, 11->14, 14->16
		visit_cnt = eff.visits_on_node_17(seq_id, efficiency)
		
		for k in range(17):

			if distance == 2:

				if k == 11:
					os_conf[seq_index].append(cm[2][k])
					os_score[seq_index].append(cm[3][k])
				elif k == 6:
					ic_conf[seq_index].append(cm[2][k])
					ic_score[seq_index].append(cm[3][k])
				elif k == 5 or k == 4:
					ic_conf[seq_index].append(cm[2][k] + cm[2][k - 2] + cm[2][k - 4])
					ic_score[seq_index].append(cm[3][k] + cm[3][k - 2] + cm[3][k - 4])

			else:
			
				# d1 (type 1)
				if k == 14:
					d1_conf[seq_index].append(__validate__(cm[2][k], visit_cnt[5]))
					d1_score[seq_index].append(__validate__(cm[3][k], visit_cnt[5]))
				# d2 (type 2)
				elif k == 11:
					d2_conf[seq_index].append(__validate__(cm[2][k], visit_cnt[4]))
					d2_score[seq_index].append(__validate__(cm[3][k], visit_cnt[4]))
				#d3 (type 3)
				elif k == 10 or k == 8: 
					d3_conf[seq_index].append(np.mean([__validate__(cm[2][10], visit_cnt[3]), \
													__validate__(cm[2][9], visit_cnt[2]), \
													__validate__(cm[2][8], visit_cnt[1]), \
													__validate__(cm[2][7], visit_cnt[0])]))
													
					d3_score[seq_index].append(np.mean([__validate__(cm[3][10], visit_cnt[3]), \
													__validate__(cm[3][9], visit_cnt[2]), \
													__validate__(cm[3][8], visit_cnt[1]), \
													__validate__(cm[3][7], visit_cnt[0])]))

	return d3_conf, d2_conf, d1_conf, d3_score, d2_score, d1_score

#
def visits_on_each_node(seqid, is_efficiency):

	return eff.visits_on_node_17(seqid, is_efficiency)
	
def set_plt_metric(plt_metric_new):

	plt_metric = plt_metric_new
		
