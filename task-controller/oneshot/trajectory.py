import numpy as np
import matplotlib.pyplot as plt

#
# data
#
'''
x_max = [0, 1, 2, 3, 4, 2, 2, 1, 0, 3, 0, 2, 1, 0, 1, 3, 4, 2, 0, 3] 
x_no = [4, 1, 3, 2, 0, 4, 3, 2, 0, 1, 4, 4, 0, 0, 3, 3, 2, 2, 1, 2, 1, 4, 0]
x_min = [3, 1, 4, 4, 4, 4, 1, 3, 0, 2, 2, 0, 1, 3, 4, 4, 0, 2, 0, 2, 2, 2, 2]
'''

x_max = [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1] 
x_no = [1, 0, 2, 1, 0, 0, 2, 1, 0, 0, 0, 2, 2, 0, 2, 2, 1, 2, 1, 1, 0, 1, 0, 1, 2, 2]
x_min = [0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]

y_max = [i for i in range(len(x_max))]
y = [i for i in range(len(x_no))]


#
# rendering
#

N = 3

xticks = ('S1-O1', 'S2-O2', 'S3-O1', 'S4-O2', 'S5-O3' )
yheight = 25

if N == 3:
	yheight = 27
	xticks = ('S1-O1', 'S2-O1', 'S3-O2')

fig = plt.figure()
fig.suptitle('Node Visit Pattern:\nTrajectories of RL agents')

# first figure
ax = fig.add_subplot(131)
ax.set_title('Max Oneshot')

plt.plot(x_max, y_max, color = 'steelblue', marker = 'o')

plt.ylabel('Steps')
plt.ylim(-1, yheight)
plt.xticks(np.arange(N), xticks)
plt.grid()

# second figure
ax = fig.add_subplot(132)
ax.set_title('No Oneshot')

plt.plot(x_no, y, color = 'forestgreen', marker = 'o')

plt.ylim(-1, yheight)
plt.xticks(np.arange(N), xticks)
plt.grid()

# second figure
ax = fig.add_subplot(133)
ax.set_title('Min Oneshot')

plt.plot(x_min, y, color = 'darkorange', marker = 'o')

plt.ylim(-1, yheight)
plt.xticks(np.arange(N), xticks)
plt.grid()

plt.show()

