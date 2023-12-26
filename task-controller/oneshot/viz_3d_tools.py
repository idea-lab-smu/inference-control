'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def scatter_3d(points):

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	n = len(points)

	# For each set of style and range settings, plot n random points in the box
	# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
	#for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
	for pt in points:
		ax.scatter(pt[0], pt[1], pt[2], c='b', marker='^')

	ax.set_xlabel('Stimuli 1')
	ax.set_ylabel('Stimuli 2')
	ax.set_zlabel('Stimuli 3')

	plt.show()
