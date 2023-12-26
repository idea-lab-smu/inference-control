import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats



'''
colors we use
'''
c_bayes = '#1f77b4'
c_maxos = '#d62728'
c_minos = '#2ca02c'
c_rand = '#ff7f0e'
	
'''
'''

def sem(data):

	ret = 0;

	#data_len = len(data)
	#data_len = np.count_nonzero(~np.isnan(data))
	#ret = np.nanstd(data) / math.pow(data_len, 0.5)

	ret = stats.sem(data, nan_policy = 'omit')

	return ret

	
def draw_os_index(data, n_subj, useRandom = False):

	plt.rcParams["figure.figsize"] = (14.0, 4.0)

	x = np.array(data).T
	xticks = range(1, n_subj + 1)
	
	fig, ax = plt.subplots()
	
	index = np.arange(n_subj)
	bar_width = 0.2

	opacity = 1
	error_config = {'ecolor': '0.3'}
	


	rects1 = ax.bar(index, x[0], bar_width,
					alpha=opacity, color=c_bayes, label='Bayesian')
					
	rects2 = ax.bar(index + bar_width * 1, x[1], bar_width,
					alpha=opacity, color=c_maxos, label='Max-OS')
					
	rects3 = ax.bar(index + bar_width * 2, x[2], bar_width,
					alpha=opacity, color=c_minos, label='Min-OS')
					
	if useRandom == True:
		rects4 = ax.bar(index + bar_width * 3, x[3], bar_width,
						alpha=opacity, color=c_rand, label='Random')


	ax.set_xlabel('Subject')
	ax.set_ylabel('One-shot index')
	ax.set_title('One-shot index by the subject ')
	ax.set_xticks(index + bar_width)
	ax.set_xticklabels(xticks)
	ax.legend()

	fig.tight_layout()
	plt.show()
	
	
def draw_os_index_minmax(data, n_subj):

	plt.rcParams["figure.figsize"] = (14.0, 4.0)

	x = np.array(data).T
	xticks = range(1, n_subj + 1)
	
	fig, ax = plt.subplots()
	
	index = np.arange(n_subj)
	bar_width = 0.2

	opacity = 1
	error_config = {'ecolor': '0.3'}

	#rects1 = ax.bar(index, x[0], bar_width,
	#				alpha=opacity, color='b', label='Bayesian')
					
	rects2 = ax.bar(index + bar_width * 1, x[1], bar_width,
					alpha=opacity, color=c_maxos, label='Max-OS')
					
	rects3 = ax.bar(index + bar_width * 2, x[2], bar_width,
					alpha=opacity, color=c_minos, label='Min-OS')
					
	#rects4 = ax.bar(index + bar_width * 3, x[3], bar_width,
	#				alpha=opacity, color='g', label='Random')


	ax.set_xlabel('Subject')
	ax.set_ylabel('One-shot index')
	ax.set_title('One-shot index by the subject ')
	ax.set_xticks(index + bar_width)
	ax.set_xticklabels(xticks)
	ax.legend()

	fig.tight_layout()
	plt.show()
	

def draw_os_index_total(data, useRandom = False):

	plt.rcParams["figure.figsize"] = (3.6, 3.6)

	x = np.array(data).T
	xticks = ['Bayesian', 'Max-OS', 'Min-OS']
	if useRandom == True:
		xticks.append('Random')
	
	avg = []
	err = []
	cnt = 0
		
	for item in x:
	
		if useRandom == False and cnt >= 3:
			break
			
		avg.append(np.mean(item))
		err.append(sem(item))

		cnt += 1
	
	x_pos = [i for i, _ in enumerate(xticks)]
	bar_color = 'tab:gray'

	plt.bar(x_pos, avg, color = bar_color, width = 0.7, yerr = err)
	plt.axhline(y = 0.5, linestyle = '--', linewidth = 0.5, color = 'k')
	
	#plt.xlabel("Optimized sequence")
	plt.ylabel("Novel pair / Non-novel pair")
	plt.ylim(0.1, 0.55)
	#plt.title("One-shot index by optimized sequence")

	plt.xticks(x_pos, xticks)

	plt.tight_layout()
	plt.show()