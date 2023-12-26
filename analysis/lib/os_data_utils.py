import scipy.io as sci		# in order to open .m files 
import numpy as np			# for the numerical operations
import math
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

import score_efficiency as eff


#
# Definition
#
filename = '../20180409-complex-all/sbj%d_node17.mat'

col_rating = 'ans_save_tot'
col_round = 'trial_info'
col_trial_detail = 'trial_info_detail'

#
# functions
#

def sem(data):

	ret = 0;
	data_len = len(data)

	ret = np.std(data) / math.pow(data_len, 0.5)

	return ret


def get_mat_data(index):

	sbj_file = filename % (index + 1)
	print sbj_file
	
	mat_data = sci.loadmat(sbj_file)

	sequence = mat_data[col_round]
	sequence_id = mat_data[col_trial_detail]
	answer = mat_data[col_rating]

	return sequence, sequence_id, answer


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
	print t_stat, p_val
	pval_buf.append(p_val)

	t_stat, p_val = stats.ttest_ind(data[0], data[2], equal_var=False)
	print t_stat, p_val
	pval_buf.append(p_val)

	t_stat, p_val = stats.ttest_ind(data[1], data[2], equal_var=False)
	print t_stat, p_val
	pval_buf.append(p_val)

	return pval_buf


def paired_t_test(data):

	pval_buf = []
	
	# t_stat, p_val = stats.mannwhitneyu(os[0], os[1])
	# p_val = p_val * 2
	t_stat, p_val = stats.ttest_rel(data[1], data[0])
	print t_stat, p_val
	pval_buf.append(p_val)
	
	t_stat, p_val = stats.ttest_rel(data[1], data[2])
	print t_stat, p_val
	pval_buf.append(p_val)
	
	'''
	t_stat, p_val = stats.ttest_rel(data[1], data[3])
	print t_stat, p_val
	pval_buf.append(p_val)
	'''
	
	t_stat, p_val = stats.ttest_rel(data[0], data[2])
	print t_stat, p_val
	pval_buf.append(p_val)
	
	'''
	t_stat, p_val = stats.ttest_rel(data[0], data[3])
	print t_stat, p_val
	pval_buf.append(p_val)
	'''

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
				linewidth = 1.2, markersize = 8, boxpad = 0.3, fontsize = 10, color = 'k'):

	# draw a line with downticks at the ends

	plt.plot([start, end], [height] * 2, '-', color = color,\
			lw = linewidth, marker = TICKDOWN, markeredgewidth = linewidth, markersize = markersize)
	
	# draw the text (stars) with a bounding box that covers up the line
	box = dict(facecolor='1.', edgecolor = 'none', boxstyle = 'Square,pad=' + str(boxpad))
	plt.text(0.5 * (start + end), height, stars(p_value),\
			 ha = 'center', va = 'center', bbox = box, size = fontsize)


def significance(ax, data, pval, rects, rect_idx):

	#order = [[1, 0], [1, 2], [1, 3], [0, 2], [0, 3]]
	order = [[1, 0], [1, 2], [0, 2]]

	offset = 1.2
	height = max_height(data) + 1

	for pi in range(len(pval)):

		height += offset
		s_rect = rects[order[pi][0]][rect_idx]
		e_rect = rects[order[pi][1]][rect_idx]

		start = s_rect.get_x() + s_rect.get_width() / 2.0
		end = e_rect.get_x() + e_rect.get_width() / 2.0
		significance_bar(start, end, height, pval[pi])
	

def autolabel(ax, rects, x_label = ''):

	'''
	Attach a text label above each bar displaying its height
	'''
	for rect in rects:

		height = rect.get_height()
		bbox = rect.get_bbox()

		# exact values
		ax.text(rect.get_x() + rect.get_width() / 2.0, 1.05 * height, '%.2f' % round(float(height), 3), \
				ha='center', va='bottom')

		# xticks
		ax.text(rect.get_x() + rect.get_width() / 2.0, -1.0, x_label, \
				ha='center', va='bottom')


def max_height(buf):

	length = len(buf)
	max_buf = []
	for li in range(length):
		max_buf.append(np.mean(buf[li]))

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
			index_mean.append(np.mean(temp))
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

	no_os_mean = [np.mean(ic[0]), np.mean(os[0])]
	no_os_err = [sem(ic[0]), sem(os[0])]

	max_os_mean = [np.mean(ic[1]), np.mean(os[1])]
	max_os_err = [sem(ic[1]), sem(os[1])]

	min_os_mean = [np.mean(ic[2]), np.mean(os[2])]
	min_os_err = [sem(ic[2]), sem(os[2])]

	rand_os_mean = [np.mean(ic[3]), np.mean(os[3])]
	rand_os_err = [sem(ic[3]), sem(os[3])]


	# now plotting
	TN = 2
	ind = np.arange(TN)
	width = 0.275	# setting for 3 rects
	#width = 0.2	# settinf for 4 rects

	fig, ax = plt.subplots()
	ax.set_ylim(0, 16)
	ylabel = 'Confidence' if efficiency == False else 'Efficiency'
	
	rects1 = ax.bar(ind, no_os_mean, width, yerr = no_os_err, label = 'Bayesian')
	rects2 = ax.bar(ind + width * 1, max_os_mean, width, yerr = max_os_err, label = 'max oneshot')
	rects3 = ax.bar(ind + width * 2, min_os_mean, width, yerr = min_os_err, label = 'min oneshot')
	#rects4 = ax.bar(ind + width * 3, rand_os_mean, width, yerr = rand_os_err, label = 'random')

	ax.set_title('%s on S-O pair inference\nOptimised = %s' % (ylabel, title))
	ax.set_ylabel('%s' % ylabel)
	plt.xticks(ind + width * 1.5, ('Non-novel\nPairs', 'Novel\nPair'))
	ax.legend(loc = 'best')

	# annoatating the exact value of each bar
	autolabel(ax, rects1)
	autolabel(ax, rects2)
	autolabel(ax, rects3)
	#autolabel(ax, rects4)
	
	ttest_rects = [rects1, rects2, rects3] #, rects4]

	# statistical significance test & draw significance for IC (idx = 0)
	ic_pval = paired_t_test(ic)
	significance(ax, ic, ic_pval, ttest_rects, 0)

	# statistical significance test & draw significance for OS
	os_pval = paired_t_test(os)
	significance(ax, os, os_pval, ttest_rects, 1)
	
	# show!
	plt.tight_layout()
	plt.show()


def draw_trial_sem(title, trial_buf, efficiency):

	bayes_mean = [np.mean(trial_buf[0])]
	bayes_err = [sem(trial_buf[0])]

	max_os_mean = [np.mean(trial_buf[1])]
	max_os_err = [sem(trial_buf[1])]

	min_os_mean = [np.mean(trial_buf[2])]
	min_os_err = [sem(trial_buf[2])]

	rand_os_mean = [np.mean(trial_buf[3])]
	rand_os_err = [sem(trial_buf[3])]

	# now plotting
	TN = 1
	ind = np.arange(TN)
	#width = 0.275	# setting for 4 rects
	#gap = 0.03
	width = 0.1		# settinf for 3 rects
	gap = 0.05
	

	fig, ax = plt.subplots()
	ax.set_ylim(0, 16)
	ylabel = 'Confidence' if efficiency == False else 'Efficiency'
	
	rects1 = ax.bar(ind, bayes_mean, width, yerr = bayes_err)#, label = 'Bayesian')
	rects2 = ax.bar(ind + (width + gap) * 1, max_os_mean, width, yerr = max_os_err)#, label = 'max oneshot')
	rects3 = ax.bar(ind + (width + gap) * 2, min_os_mean, width, yerr = min_os_err)#, label = 'min oneshot')
	#rects4 = ax.bar(ind + (width + gap) * 3, rand_os_mean, width, yerr = rand_os_err)#, label = 'random')

	ax.set_title('%s on S-O pair inference\nOptimised = %s' % (ylabel, title))
	ax.set_ylabel('%s' % ylabel)
	ax.legend(loc = 'best')

	# annoatating the exact value of each bar
	autolabel(ax, rects1, 'Bayesian')
	autolabel(ax, rects2, 'Max OS')
	autolabel(ax, rects3, 'Min OS')
	#autolabel(ax, rects4, 'Random')

	# statistical significance test & draw significance for IC (idx = 0)
	trial_pval = paired_t_test(trial_buf)
	ttest_rects = [rects1, rects2, rects3] #, rects4]
	significance(ax, trial_buf, trial_pval, ttest_rects , 0)

	# show!
	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off')
	plt.tight_layout()
	plt.show()


def __validate__(score, visit_cnt):

	res = 0.0

	if visit_cnt > 0.0:

		res = score / visit_cnt

	return res	
	

def distinct_ic_os_buf(conf_map, efficiency, distance):

	# we see the result
	os = [[], [], [], []]	# no, max, min, random
	ic = [[], [], [], []]	# no, max, min, random

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
					os[seq_index].append(cm[2][k])
				elif k == 6:
					ic[seq_index].append(cm[2][k])
				elif k == 5 or k == 4:
					ic[seq_index].append(cm[2][k] + cm[2][k - 2] + cm[2][k - 4])

			else:
			
				if k == 14:
					os[seq_index].append(__validate__(cm[2][k], visit_cnt[5]))
				elif k == 11:
					ic[seq_index].append(__validate__(cm[2][k], visit_cnt[4]))
				elif k == 10: 
					ic[seq_index].append(__validate__(cm[2][k], visit_cnt[3]) \
										+ __validate__(cm[2][k - 1], visit_cnt[2]))
				elif k == 8: 
					ic[seq_index].append(__validate__(cm[2][k], visit_cnt[1]) \
										+ __validate__(cm[2][k - 1], visit_cnt[0]))

	return ic, os

#
def visits_on_each_node(cnt_nodes, seqid, is_efficiency):

	if cnt_nodes == 17:
	
		return eff.visits_on_node_17(seqid - 1, is_efficiency)
	
	elif cnt_nodes == 5:
	
