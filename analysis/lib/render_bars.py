'''
render_bars.py

	- 
	- 
'''
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.markers import TICKDOWN
import numpy as np
import scipy.stats as stats

#import seaborn as sns

import lib.os_data_utils_17 as odu


# accessed by set_plt_metric()
plt_metric = {'figx':4.0, 'figy':2.5}
plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
plt.rcParams.update({'font.size': 13})

# globals
g_legends = ['Bayesian+', 'oneshot+', 'oneshot-']
g_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

strconf = 'Causal rating'
strscore = 'Test score\n (normalized causal rating)'
streff = 'Efficiency\n (test score / # of presentations)'

ytop_bias = 0.0

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
		res_str = "n.s"

	#return res_str + '\n' + str_parenthesis(str(round(p, 3)))
	#return res_str + '\n' + 'p=' + str(round(p, 4))
	return res_str

def significance(data1, data2, rects, start, end, height, \
				linewidth = 1.0, markersize = 3, boxpad = 0.3, fontsize = 11, color = 'k'):

	fontsize = 10
	
	t_stat, p_val = stats.ttest_ind(data1, data2)
	star = stars(p_val)
	
	plt.plot([start, end], [height] * 2, '-', color = color,\
			lw = linewidth, marker = TICKDOWN, markeredgewidth = linewidth, markersize = markersize)
	
	box = dict(facecolor='1.', edgecolor = 'none', boxstyle = 'Square,pad=' + str(boxpad))
	plt.text(0.5 * (start + end), height, star, ha = 'center', va = 'center', bbox = box, size = fontsize)

def significance_bar(start, end, height, p_value,
				linewidth = 1.0, markersize = 3, boxpad = 0.3, fontsize = 11, color = 'k'):

	# draw a line with downticks at the ends

	plt.plot([start, end], [height] * 2, '-', color = color,
			lw = linewidth, marker = TICKDOWN, markeredgewidth = linewidth, markersize = markersize)
	
	# draw the text (stars) with a bounding box that covers up the line
	box = dict(facecolor='1.', edgecolor = 'none', boxstyle = 'Square,pad=' + str(boxpad))
	plt.text(0.5 * (start + end), height, stars(p_value),
			 ha = 'center', va = 'center', bbox = box, size = fontsize)

def significance_rect_obj(data, pval, rect_objs, height = 7.5, offset = 0.5, useRandom = False):

	if useRandom == True:
		order = [[1, 0], [1, 2], [1, 3], [0, 2], [0, 3]]
	else:
		order = [[1, 0], [1, 2], [0, 2]]

	#offset = plt_metric['step']
	#offset = 0.5
	#height = odu.max_height(data) + (offset * 2.0)

	for pi in range(len(pval)):

		height += offset
		s_rect = rect_objs[order[pi][0]]
		e_rect = rect_objs[order[pi][1]]

		start = s_rect.get_x() + s_rect.get_width() / 2.0
		end = e_rect.get_x() + e_rect.get_width() / 2.0
		significance_bar(start, end, height, pval[pi])


def plot_bars(avg, err, ic_pairs, os_pairs, color, label):

	rects = plt.bar(ind, avg, width, color = color, yerr = err)
	
	plt.title(label)
	
	significance(ic_pairs, os_pairs, rects, 0.0, 1.0, ytop - 0.5)


def show_kb_learning(ic, os, title, efficiency):

	useRandom = False
	
	plt_metric = {'figx':6.0, 'figy':3.5}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
	
	nitems = len(ic)

	ind = np.arange(2)
	xticks = [2, 1]
	width = 0.8

	iplot = 0
	if title == 'Confidence':
		ylabel = strconf if efficiency == False else streff
		ytop = 9.0 + ytop_bias if efficiency == False else 4.0 + ytop_bias
		#ytop = 12.0 + ytop_bias if efficiency == False else 4.0 + ytop_bias
	else:
		ylabel = strscore if efficiency == False else streff
		ytop = 10.0 + ytop_bias if efficiency == False else 4.0 + ytop_bias
		#ytop = 12.0 + ytop_bias if efficiency == False else 4.0 + ytop_bias
	
	offset = ytop * 0.07
	
	fig = plt.figure()
	
	# i = 0 bayes+; 1 oneshot+; 2 oneshot-; 3 random
	for i in range(nitems - 1):
		
		color = g_colors[i]
		label = g_legends[i]
		
		ic_pairs = ic[i]
		os_pairs = os[i]
		avg = [np.nanmean(ic_pairs), np.nanmean(os_pairs)]
		err = [odu.sem(ic_pairs), odu.sem(os_pairs)]

		plt.subplot(1, 3, i + 1)
		
		rects = plt.bar(ind, avg, width, color = color, yerr = err)
		plt.title(label)
		plt.xticks(ind, xticks)
		plt.ylim(0, ytop)
		
		if i < 1:
			plt.ylabel(ylabel)
		
		significance(ic_pairs, os_pairs, rects, 0.0, 1.0, ytop - offset)
		
	fig.text(0.33, 0.01, 'Degree of S-O pair nodes')
	
	plt.tight_layout()
	plt.show()

# plotting the confidence and score on the S-O pairs having the lowest degree; across the conditions
def show_kb_learning_across_condition(ic, os, title, efficiency):

	useRandom = False
	
	plt_metric = {'figx':3.5, 'figy':3.5}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
	
	ind = np.arange(3)
	xticks = g_legends
	width = 0.8

	if title == 'Confidence':
		ylabel = strconf if efficiency == False else streff
		ytop = 9.0 + ytop_bias if efficiency == False else 4.0 + ytop_bias
	else:
		ylabel = strscore if efficiency == False else streff
		ytop = 8.0 + ytop_bias if efficiency == False else 4.0 + ytop_bias
	
	offset = ytop * 0.07
	
	# i = 0 bayes+; 1 oneshot+; 2 oneshot-; 3 random
	avg = [np.nanmean(os[0]), np.nanmean(os[1]), np.nanmean(os[2])]
	err = [odu.sem(os[0]), odu.sem(os[1]), odu.sem(os[2])]
	
	fig, ax = plt.subplots()

	rects = plt.bar(ind, avg, width, color = 'tab:gray', yerr = err)
	#plt.title(label)
	plt.xticks(ind, xticks, rotation = 30)
	plt.ylim(0, ytop)
	plt.ylabel(ylabel)
	
	#significance(ic_pairs, os_pairs, rects, 0.0, 1.0, ytop - 0.5)
	F_statistic, pVal = stats.f_oneway(os[0], os[1], os[2])
	print('oneway ANOVA : F={0:.1f}, p={1}'.format(F_statistic, pVal))
	if pVal < 0.05:
		print('Significant!')
		
	os_pval = odu.paired_t_test(os, False)
	significance_rect_obj(os, os_pval, rects, height = ytop - (offset * 3.8), offset = offset, useRandom = False)
		
	plt.tight_layout()
	plt.show()
	
def downsize(buf, inc):

	N = len(buf)
	ret_buf = []
	
	for i in range(0, N, inc):
		ret_buf.append((buf[i] + buf[i + 1]) / 2.0)
		
	return ret_buf
	
def show_kb_learning_difficult(d3, d2, d1, title, efficiency):

	useRandom = False
	
	plt_metric = {'figx':9.0, 'figy':3.5}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
	
	xticks = ['[4, 6)','[2, 4)','[0, 2)']

	nitems = len(d3)
	ind = np.arange(len(xticks))
	width = 0.8

	iplot = 0
	ytop = 8.0
	ybottom = 0
	
	if title == 'Confidence':
		ylabel = strconf if efficiency == False else streff
		ytop = 8.0 if efficiency == False else 4.0
	else:
		ylabel = strscore if efficiency == False else streff
		ytop = 3.0 if efficiency == False else 2.0
	
	offset = ytop * 0.07
	
	fig = plt.figure()
	#fig.suptitle('This is a somewhat long figure title')
	
	# i = 0 bayes+; 1 oneshot+; 2 oneshot-; 3 random
	for i in range(nitems - 1):
		
		color = g_colors[i]
		label = g_legends[i]
		
		d3_pairs = d3[i]
		d2_pairs = d2[i]
		d1_pairs = d1[i]
		avg = [np.nanmean(d3_pairs), np.nanmean(d2_pairs), np.nanmean(d1_pairs),]
		err = [odu.sem(d3_pairs), odu.sem(d2_pairs), odu.sem(d1_pairs)]

		plt.subplot(1, 3, i + 1)
		
		rects = plt.bar(ind, avg, width, color = color, yerr = err)
		plt.title(label)
		plt.xticks(ind, xticks)
		plt.ylim(0, ytop)
		
		if i < 1:
			plt.ylabel(ylabel)
		
		#significance(ic_pairs, os_pairs, rects, 0.0, 1.0, ytop - 0.5)
		d3_new = downsize(d3_pairs, 2)
		dn_pairs = [d3_new, d2_pairs, d1_pairs]
		
		F_statistic, pVal = stats.f_oneway(d3_new, d2_pairs, d1_pairs)
		print('oneway ANOVA : F={0:.1f}, p={1}'.format(F_statistic, pVal))
		if pVal < 0.05:
			print('Significant!')
		
		dn_pval = odu.paired_t_test(dn_pairs, False)
		significance_rect_obj(dn_pairs, dn_pval, rects, \
					height = ytop - (offset * 3.8), offset = offset, useRandom = False)
		
	fig.text(0.33, 0.01, 'Degree of S-O pair nodes')
	
	plt.tight_layout()
	plt.show()

# plotting the confidence and score on the S-O pairs having the lowest degree; across the conditions
def show_kb_learning_across_condition_difficult(d3, d2, d1, title, efficiency):

	useRandom = False
	
	plt_metric = {'figx':3.5, 'figy':3.5}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])
	
	ind = np.arange(3)
	xticks = g_legends
	width = 0.8

	iplot = 0
	if title == 'Confidence':
		ylabel = strconf if efficiency == False else streff
		ytop = 8.0 if efficiency == False else 4.0
	else:
		ylabel = strscore if efficiency == False else streff
		ytop = 3.0 
	
	offset = ytop * 0.07
	
	# i = 0 bayes+; 1 oneshot+; 2 oneshot-; 3 random
	avg = [np.nanmean(d1[0]), np.nanmean(d1[1]), np.nanmean(d1[2])]
	err = [odu.sem(d1[0]), odu.sem(d1[1]), odu.sem(d1[2])]
	
	fig, ax = plt.subplots()

	rects = plt.bar(ind, avg, width, color = 'tab:gray', yerr = err)
	#plt.title(label)
	plt.xticks(ind, xticks, rotation = 30)
	plt.ylim(0, ytop)
	plt.ylabel(ylabel)
	
	#significance(ic_pairs, os_pairs, rects, 0.0, 1.0, ytop - 0.5)
	F_statistic, pVal = stats.f_oneway(d1[0], d1[1], d1[2])
	print('oneway ANOVA : F={0:.1f}, p={1}'.format(F_statistic, pVal))
	if pVal < 0.05:
		print('Significant!')
	
	d1_pval = odu.paired_t_test(d1, False)
	significance_rect_obj(d1, d1_pval, rects, \
						height = ytop - (offset * 3.8), offset = offset, useRandom = False)
		
	plt.tight_layout()
	plt.show()
