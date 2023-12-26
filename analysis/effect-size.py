import math
import scipy.stats as stats
import scipy.spatial.distance as dist
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

def draw(effsize, xlabel = '', ylabel = '', title = '', xticks = [], ylim = [0.0, 1.0], width = 0.8, figconf =  {'figx':3.6, 'figy':3.6}):

	plt_metric = figconf
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])

	ind = np.arange(len(effsize))
	xticks = xt
	ylabel = yl
	
	plt.title(title, size = 13)
	plt.xticks(ind, xticks)
	plt.xlabel(xl, size = 13)
	plt.ylabel(ylabel, size = 13)
	plt.ylim(ylim[0], ylim[1])

	rects = plt.bar(xt, effsize, width, color = 'gray')
	
	# show!
	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=True,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=True)

	plt.tight_layout()
	plt.show()
	
if __name__ == '__main__':

	# these effect size numbers are coming from each analysis script
	# e.g., 5-trial or 17-trial family...
	
	'''
	
	# test score
	xt = ['Easy', 'Medium', 'Difficult']
	xl = 'Level of difficulty'
	yl = 'Hedge\'s g\n(Oneshot+ - Oneshot-)'
	title = 'Effect size of the task control\n(Test score)'
	
	#effsize = [0.452, 0.361, 0.016]
	effsize = [0.240, 0.307, 0.016]
	draw(effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.01, 0.35])
	
	# efficiency
	title = 'Effect size of the task control\n(Efficiency)'
	
	#eff_effsize = [0.207, 0.154, 0.178]
	eff_effsize = [0.158, 0.134, 0.178]
	draw(eff_effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.1, 0.2])
		
	# osidx_u50
	xt = ['High', 'Low']
	xl = 'Oneshot bias'
	yl = 'Hedge\'s g\n(Oneshot+ - Oneshot-)'
	title = 'Effect size of the \ntask control (Test score)'
	figconf = {'figx':3.6, 'figy':3.6}
	
	u50_effsize = [0.563, 0.356]
	draw(u50_effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.3, 0.6], width = 0.6, figconf = figconf)

	yl = 'Hedge\'s g\n(Oneshot+ - Uniform)'
	figconf = {'figx':3.6, 'figy':3.6}	
	
	u50_effsize = [0.199, 0.099]
	draw(u50_effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.0, 0.21], width = 0.6, figconf = figconf)
	
	# efficiency
	xt = ['High', 'Low']
	xl = 'Oneshot bias'
	yl = 'Hedge\'s g\n(Oneshot+ - Oneshot-)'
	title = 'Effect size of the \ntask control (Efficiency)'
	figconf = {'figx':3.6, 'figy':3.6}
	
	u50_effsize = [0.297, 0.122]
	draw(u50_effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.1, 0.31], width = 0.6, figconf = figconf)

	yl = 'Hedge\'s g\n(Oneshot+ - Uniform)'
	figconf = {'figx':3.6, 'figy':3.6}	
	
	u50_effsize = [0.485, 0.344]
	draw(u50_effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.33, 0.5], width = 0.6, figconf = figconf)
	
	# effect size - high, mid, low
	xt = ['High', 'Mid', 'Low']
	xl = 'Oneshot bias'
	yl = 'Hedge\'s g\n(Oneshot+ - Oneshot-)'
	title = 'Effect size of the \ntask control (Test score)'
	figconf = {'figx':3.6, 'figy':3.6}
	
	effsize = [0.615, 0.498, 0.279]
	draw(effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.25, 0.625], figconf = figconf)
	
	xt = ['High-Low', 'High-Mid', 'Mid-Low']
	xl = 'Control condition (Oneshot+)'
	yl = 'Hedge\'s g\n(among High, mid low)'
	title = 'Effect size of the \ntask control (Test score)'
	figconf = {'figx':3.6, 'figy':3.6}
	
	effsize = [0.406, 0.331, 0.078]
	draw(effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [0.05, 0.425], figconf = figconf)
	'''
	#######
	
	# test score (Oneshot+ vs Uniform)
	xt = ['Easy', 'Medium', 'Difficult']
	xl = 'Level of difficulty'
	yl = 'Hedge\'s g\n(Oneshot+ - Uniform)'
	title = 'Effect size of the task control\n(Test score)'
	
	effsize = [0.245, 0.564, -0.026]
	draw(effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [-0.1, 0.65])
	
	# efficiency
	title = 'Effect size of the task control\n(Efficiency)'
	
	#eff_effsize = [0.207, 0.154, 0.178]
	eff_effsize = [0.439, 0.622, 0.073]
	draw(eff_effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [-0.1, 0.65])
	
	# test score (Oneshot- vs Uniform)
	xt = ['Easy', 'Medium', 'Difficult']
	xl = 'Level of difficulty'
	yl = 'Hedge\'s g\n(Oneshot- - Uniform)'
	title = 'Effect size of the task control\n(Test score)'
	
	#effsize = [0.452, 0.361, 0.016]
	effsize = [-0.028, 0.210, -0.032]
	draw(effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [-0.1, 0.65])
	
	# efficiency
	title = 'Effect size of the task control\n(Efficiency)'
	
	#eff_effsize = [0.207, 0.154, 0.178]
	eff_effsize = [0.185, 0.335, -0.048]
	draw(eff_effsize, xlabel = xl, ylabel = yl, title = title, xticks = xt, ylim = [-0.1, 0.65])
