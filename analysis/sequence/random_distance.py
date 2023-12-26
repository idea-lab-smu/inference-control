import numpy as np
import math
import scipy.stats as stats
import scipy.spatial.distance as dist
import random

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN


#
# sets of optimised sequences

seq_bayes = [
[11, 5, 1, 0, 5, 15, 4, 3, 7, 9, 6, 14, 2, 11, 7, 15, 10, 8, 14, 9, 12, 13, 17, 4, 16, 0, 1, 3, 10, 12, 17, 6, 16, 2, 8, 13],
[10, 0, 16, 7, 12, 5, 11, 9, 6, 15, 4, 3, 17, 2, 14, 1, 8, 13, 5, 0, 1, 16, 4, 17, 14, 9, 7, 6, 15, 11, 10, 13, 12, 2, 3, 8],
[10, 7, 8, 16, 4, 9, 12, 11, 2, 5, 17, 0, 3, 6, 14, 1, 15, 13, 10, 12, 8, 13, 9, 4, 5, 11, 3, 15, 2, 1, 6, 0, 16, 14, 7, 17]
]

seq_maxos = [
[15, 15, 3, 5, 2, 7, 12, 1, 13, 12, 7, 9, 15, 12, 13, 12, 6, 4, 2, 1, 8, 13, 10, 8, 12, 17, 0, 17, 11, 6, 14, 14, 12, 7, 10, 16],
[1, 10, 8, 7, 3, 11, 4, 8, 12, 0, 1, 2, 10, 0, 4, 16, 13, 11, 17, 1, 7, 4, 8, 6, 4, 11, 13, 15, 5, 15, 0, 7, 8, 13, 14, 9],
[1, 1, 11, 6, 13, 2, 6, 12, 2, 12, 2, 0, 16, 15, 6, 5, 6, 12, 0, 8, 11, 7, 15, 15, 12, 3, 2, 15, 9, 14, 2, 9, 10, 16, 17, 3, 4]
]

seq_minos = [
[4, 15, 15, 4, 15, 8, 15, 8, 16, 1, 6, 12, 16, 4, 15, 15, 10, 9, 14, 6, 15, 6, 11, 12, 14, 8, 10, 7, 9, 4, 1, 10, 14, 11, 1, 15],
[8, 9, 6, 14, 5, 0, 10, 5, 16, 5, 12, 7, 15, 5, 4, 9, 15, 8, 17, 7, 8, 5, 5, 2, 11, 5, 6, 11, 14, 12, 7, 11, 4, 0, 12, 0],
[7, 15, 11, 16, 2, 12, 15, 15, 15, 12, 15, 15, 15, 15, 0, 8, 1, 16, 10, 9, 6, 2, 9, 7, 15, 3, 4, 7, 9, 16, 10, 1, 13, 14, 12, 10]
]

seq_random = [
[5, 14, 0, 11, 10, 16, 12, 0, 11, 1, 8, 7, 16, 6, 0, 13, 12, 1, 17, 11, 1, 7, 10, 7, 5, 11, 10, 12, 16, 5, 13, 4, 10, 6, 9, 9],
[3, 15, 13, 4, 5, 5, 8, 0, 2, 16, 13, 9, 11, 11, 7, 8, 2, 9, 17, 9, 6, 0, 2, 13, 14, 2, 8, 11, 2, 5, 2, 3, 16, 16, 15, 12],
[2, 11, 0, 0, 2, 5, 9, 10, 3, 8, 13, 1, 0, 2, 17, 3, 3, 0, 8, 4, 13, 3, 0, 5, 17, 12, 6, 1, 6, 17, 3, 7, 8, 12, 12, 3]
]

seq_bayes_5 = [
[2, 0, 1, 3, 4, 2, 4, 0, 3, 1, 2, 0, 4, 3, 4, 3, 3, 2, 0, 1],
[1, 2, 3, 0, 4, 1, 2, 0, 3, 4, 4, 4, 3, 0, 2, 4, 3, 1, 0, 2],
[1, 2, 3, 0, 4, 1, 0, 2, 3, 4, 1, 0, 0, 4, 2, 3, 4, 0, 2, 1]
]

seq_maxos_5 = [
[0, 3, 2, 1, 4, 1, 1, 0, 2, 2, 3, 0, 3, 2, 0, 0, 1, 4, 2, 3],
[3, 1, 2, 3, 4, 0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 2, 1, 4, 3, 2],
[3, 3, 2, 1, 4, 0, 0, 0, 1, 1, 2, 2, 3, 0, 2, 0, 1, 4, 3, 2]
]

seq_minos_5 = [
[0, 2, 2, 1, 0, 0, 2, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3],
[4, 1, 0, 2, 3, 1, 2, 0, 2, 3, 1, 3, 1, 2, 0, 0, 3, 0, 3, 3],
[0, 2, 4, 1, 3, 2, 0, 2, 3, 1, 3, 2, 1, 0, 2, 0, 3, 3, 3, 3]
]

seq_random_5 = [
#1
[1,3,3,4,4,0,2,4,1,0,2,3,3,2,4,0,1,2,1,0],
[4,1,1,2,1,1,0,0,3,0,2,3,3,2,3,2,0,4,4,4],
[1,0,1,3,2,0,3,2,3,4,4,1,3,0,2,4,0,2,4,1],
#2
[0,0,1,3,1,1,1,4,2,0,3,2,3,2,3,2,4,0,4,4],
[4,3,0,2,0,2,4,1,1,3,2,3,1,4,0,2,4,3,0,1],
[0,0,4,0,2,3,3,3,2,1,1,2,4,3,2,0,1,1,4,4],
#3
[4,2,1,1,4,4,0,3,2,0,0,3,2,0,3,2,3,4,1,1],
[3,0,0,4,1,2,2,3,1,3,4,0,1,4,2,1,0,4,2,3],
[0,3,4,1,3,0,2,4,4,2,1,0,3,1,2,3,0,1,2,4],
#4
[4,3,3,0,0,2,4,1,2,2,4,1,4,3,3,1,0,2,1,0],
[2,4,2,1,3,0,3,4,4,2,0,0,1,2,4,0,3,1,1,3],
[0,2,1,2,3,4,3,4,2,1,1,3,4,0,0,1,3,2,0,4],
#5
[1,2,4,3,2,2,0,0,4,3,2,0,3,4,3,0,4,1,1,1],
[1,4,4,0,4,1,1,3,0,2,3,2,4,3,0,2,3,1,0,2],
[4,1,0,4,0,3,4,2,1,3,1,3,2,4,2,1,2,0,0,3],
#6
[2,1,1,3,0,0,0,2,0,4,4,1,3,1,3,4,2,3,2,4],
[3,0,4,3,2,4,2,0,4,3,1,1,4,2,1,0,3,1,0,2],
[4,3,2,2,0,1,4,0,1,2,3,0,4,2,3,1,1,4,0,3],
#7
[0,2,3,1,0,4,4,1,0,3,2,1,3,2,0,1,2,4,3,4],
[2,2,3,0,3,1,2,0,4,4,1,1,4,0,3,3,0,2,1,4],
[2,1,2,0,3,0,1,0,2,1,3,2,4,3,1,3,4,4,4,0],
#8
[1,0,3,0,1,3,4,3,2,0,2,4,2,1,2,3,1,4,4,0],
[4,1,3,1,3,1,0,1,4,2,0,3,0,2,0,4,2,4,3,2],
[2,1,2,0,4,2,1,0,3,1,3,3,0,0,4,1,3,4,2,4],
#9
[0,4,0,3,1,4,3,2,3,4,3,2,2,1,1,4,2,1,0,0],
[3,3,4,1,0,4,4,1,1,2,0,0,4,3,3,2,0,1,2,2],
[1,0,3,3,4,0,4,2,3,4,3,1,2,0,1,0,4,2,2,1],
#10
[0,2,1,2,3,2,3,3,4,2,4,0,4,0,4,1,1,3,0,1],
[4,0,1,0,2,4,2,3,3,0,3,2,3,4,4,2,0,1,1,1],
[1,4,0,2,0,1,3,1,3,2,4,0,0,4,2,2,3,3,1,4],
#11
[2,3,3,1,4,2,4,1,0,0,4,4,3,2,0,1,0,2,3,1],
[3,3,4,2,2,4,1,1,0,2,0,0,4,3,3,4,2,1,1,0],
[1,2,3,2,1,4,3,4,4,4,2,1,0,1,2,3,0,0,3,0],
#12
[0,3,0,4,0,1,2,1,0,2,4,2,3,4,3,3,1,4,1,2],
[2,4,3,0,2,2,1,4,1,1,2,4,3,4,0,3,1,3,0,0],
[1,3,3,0,4,4,1,3,2,2,4,4,1,0,0,2,3,0,1,2],
#13
[1,1,2,4,3,2,3,4,4,0,4,3,0,0,0,2,1,3,1,2],
[3,4,2,1,3,0,4,2,1,1,0,1,0,3,4,4,2,2,0,3],
[3,1,2,3,3,2,0,1,2,0,4,0,1,3,2,0,4,1,4,4],
#14
[0,3,0,3,2,2,1,3,3,0,4,1,4,2,0,4,1,4,1,2],
[1,4,3,4,3,3,2,0,2,4,4,0,2,1,0,3,1,2,1,0],
[3,0,1,2,1,2,0,3,4,1,0,4,2,3,3,2,4,1,0,4],
#15
[0,3,2,4,3,3,1,2,3,4,0,0,1,2,1,4,4,1,0,2],
[1,1,3,2,0,1,1,3,4,4,0,3,0,4,3,2,2,2,0,4],
[4,0,1,1,3,0,0,1,2,2,1,4,4,2,3,3,4,0,3,2],
#16
[2,2,1,0,2,4,1,1,4,0,0,0,3,2,3,3,4,1,4,3],
[4,1,3,0,1,1,4,3,2,1,4,3,0,2,3,4,0,2,2,0],
[0,1,1,4,2,4,0,0,3,1,3,2,1,3,2,4,2,0,3,4],
#17
[2,1,3,3,0,4,0,2,1,4,0,1,3,0,1,4,3,2,4,2],
[2,1,0,0,1,4,2,3,2,3,2,3,4,0,1,0,1,4,4,3],
[4,1,4,1,2,0,0,3,4,3,3,2,1,1,2,0,0,4,2,3],
#18
[4,4,1,0,3,3,3,0,1,4,1,2,4,1,2,0,2,2,0,3],
[4,2,4,1,0,3,0,3,0,1,3,2,4,2,0,1,1,3,2,4],
[4,3,1,4,0,2,4,2,2,1,3,1,0,4,1,0,0,3,3,2],
#19
[3,4,4,1,1,1,0,3,3,0,2,2,0,1,0,2,4,2,3,4],
[3,2,2,2,1,1,4,4,1,0,2,3,3,0,0,3,1,0,4,4],
[2,2,2,0,1,3,4,4,1,3,3,0,4,2,1,3,4,0,1,0],
#20
[3,0,1,0,4,4,4,1,1,2,2,3,3,2,2,1,4,3,0,0],	
[0,2,0,1,0,2,4,4,1,3,4,3,3,4,0,2,1,2,1,3],
[3,0,3,3,3,1,2,4,2,4,0,1,4,2,0,4,2,1,1,0],
#21
[4,0,2,0,0,3,2,1,4,2,3,4,0,2,1,1,1,4,3,3],	
[0,4,0,3,1,4,2,3,3,1,0,2,1,2,2,4,0,1,4,3],
[4,3,1,3,4,2,0,2,0,4,1,2,0,2,4,1,1,3,0,3],
#22
[4,2,3,2,2,3,0,1,0,4,3,0,0,1,3,4,4,1,2,1],	
[2,2,1,3,4,1,0,4,0,3,0,2,2,1,4,3,0,1,3,4],
[4,4,0,0,2,3,2,4,2,2,4,1,1,1,1,0,3,3,0,3],
#23
[4,3,0,3,1,2,3,4,0,4,2,4,1,3,1,2,0,0,1,2],
[1,3,4,0,2,1,3,1,3,2,0,1,2,2,0,4,3,4,4,0],	
[1,4,4,2,4,0,1,2,0,1,3,3,2,3,1,0,4,2,3,0],
#24
[2,0,4,3,3,1,0,1,2,2,0,2,4,3,0,1,4,1,4,3],	#1
[0,4,3,1,4,2,0,3,0,4,3,2,2,2,0,1,1,3,1,4],
[4,4,3,0,3,1,1,1,4,0,2,3,0,2,4,1,2,2,3,0]
]

'''
seq_random_5 = [
[4, 3, 2, 0, 0, 1, 0, 2, 1, 4, 2, 4, 2, 3, 4, 3, 1, 1, 0, 3], 
[2, 3, 2, 1, 4, 0, 4, 4, 0, 1, 3, 3, 4, 0, 1, 3, 2, 0, 2, 1], 
[1, 0, 2, 3, 0, 0, 1, 3, 3, 1, 2, 4, 1, 0, 4, 3, 4, 2, 2, 4],
[3,0,2,2,4,1,0,1,4,4,0,1,3,3,3,4,2,1,0,2],
[2,0,4,4,1,0,2,4,1,4,3,0,3,2,3,3,1,2,0,1],
[4,2,2,3,4,4,1,0,4,1,1,0,0,1,0,2,2,3,3,3],
[3,1,3,4,2,1,2,0,4,1,4,3,2,4,3,1,2,0,0,0],
[2,2,4,1,3,2,0,3,1,3,0,2,4,4,1,0,3,1,4,0],
[4,3,2,2,4,0,2,3,4,0,1,0,1,3,1,1,4,3,0,2],
[3,1,3,4,2,1,2,0,4,1,4,3,2,4,3,1,2,0,0,0],
[2,2,4,1,3,2,0,3,1,3,0,2,4,4,1,0,3,1,4,0],
[4,3,2,2,4,0,2,3,4,0,1,0,1,3,1,1,4,3,0,2],
[1,2,0,0,0,4,1,2,2,4,3,2,3,0,1,1,4,4,3,3],
[0,3,4,0,3,3,2,2,4,4,1,0,3,4,2,1,1,2,0,1],
[2,3,4,4,0,3,4,3,0,2,0,2,1,3,0,1,1,2,1,4],
[1,3,4,4,2,4,0,1,3,3,4,1,2,1,2,0,2,0,3,0],
[2,2,3,4,0,4,3,1,2,4,0,3,4,0,3,1,0,2,1,1],
[4,2,0,0,2,1,3,2,0,4,1,3,2,1,3,1,3,4,0,4],
[2,1,3,4,4,1,0,1,2,2,0,4,4,3,0,2,3,1,3,0],
[2,0,2,0,1,1,0,3,2,3,4,4,2,3,0,4,1,1,3,4],
[1,0,3,0,4,2,2,2,3,1,0,4,2,4,4,1,3,1,3,0],
[4,1,3,1,3,4,0,4,1,0,3,0,2,0,2,4,2,1,2,3],
[2,0,0,3,4,1,1,1,2,0,4,4,0,3,3,2,3,4,2,1],
[3,3,0,2,2,1,0,4,1,1,2,2,4,3,0,4,4,3,0,1],
[3,1,3,4,1,1,0,2,0,3,0,2,1,4,4,0,2,3,4,2],
[0,1,4,4,3,2,1,1,0,0,2,2,3,0,4,4,2,3,1,3],
[0,3,3,3,2,1,2,4,0,1,1,4,0,3,0,4,2,2,4,1],
[2,3,1,4,0,3,1,4,2,1,2,4,3,0,3,1,0,4,0,2],
[3,2,0,0,0,4,1,3,1,4,2,3,2,0,1,4,3,4,2,1],
[4,4,2,0,1,1,2,2,0,1,3,0,4,3,3,1,3,0,4,2],
[1,4,3,0,4,1,4,2,0,0,2,1,1,4,2,2,3,3,3,0],
[1,3,2,2,4,3,2,0,1,1,2,0,4,3,0,4,4,1,0,3],
[3,1,3,1,4,3,4,0,0,0,0,2,1,1,4,2,2,2,3,4],
[3,4,3,4,0,3,2,0,0,1,1,2,2,4,3,0,4,1,2,1],
[2,1,2,3,0,0,2,2,1,3,0,3,4,4,4,3,1,1,0,4],
[2,4,2,2,1,0,4,4,3,4,0,3,3,1,0,0,1,3,2,1],
[1,0,4,1,1,2,2,4,1,4,3,3,3,4,2,0,2,0,0,3],
[4,0,2,0,3,2,3,4,1,0,4,2,3,1,2,0,1,1,4,3],
[2,4,0,2,0,2,2,3,3,4,3,1,0,0,1,1,1,3,4,4],
[3,1,0,1,0,3,4,2,3,1,1,0,4,2,2,4,2,4,0,3],
[2,1,2,0,1,3,2,4,0,4,3,4,4,1,2,1,3,0,3,0],
[2,1,3,0,4,1,2,2,4,1,3,4,4,1,0,3,0,2,0,3],
[0,2,3,4,4,3,1,1,1,1,4,0,4,0,2,3,0,2,2,3],
[0,3,4,2,3,3,2,1,1,3,4,4,0,2,0,2,1,1,4,0],
[2,1,2,4,3,0,3,1,0,3,4,2,4,2,1,1,3,4,0,0],
[4,3,0,1,2,1,3,4,2,4,4,2,0,0,3,1,3,0,2,1],
[0,2,0,4,2,4,2,1,4,1,0,1,0,4,3,3,2,3,3,1],
[1,4,4,3,2,0,4,0,3,3,4,2,0,1,3,0,1,2,2,1],
[2,4,1,3,3,4,0,1,1,3,3,2,4,2,2,0,4,1,0,0],
[4,3,4,1,2,1,0,2,4,4,3,2,3,3,0,0,1,0,1,2],
[3,2,0,4,1,0,1,4,3,4,4,2,1,0,2,1,3,0,3,2],
[4,3,3,4,0,2,0,2,1,0,3,1,4,1,1,2,0,4,2,3],
[2,0,3,4,1,3,3,0,2,0,3,4,1,4,1,1,0,2,4,2],
[2,4,4,3,0,0,0,4,2,3,0,1,2,1,1,1,2,4,3,3],
[3,2,2,3,1,4,1,2,3,0,2,4,1,0,3,0,4,0,4,1],
[3,4,0,0,4,3,3,3,2,1,2,0,0,1,2,4,4,1,1,2],
[3,0,1,4,1,3,4,0,4,1,3,1,2,0,2,2,0,3,4,2],
[4,3,1,0,0,1,4,2,3,1,2,0,3,0,2,4,1,4,2,3],
[4,4,3,1,3,1,1,2,1,0,0,2,0,4,0,3,2,3,2,4],
[2,4,0,4,0,2,1,1,3,1,4,0,2,4,2,0,3,3,1,3],
[1,0,3,4,3,2,4,1,4,0,3,3,4,2,1,0,0,2,2,1],
[4,2,2,3,2,3,1,0,2,0,1,4,0,1,4,3,0,4,1,3],
[4,1,0,3,2,0,1,2,1,4,4,4,3,2,0,0,2,1,3,3],
[3,4,1,3,3,4,0,2,2,2,0,1,3,4,2,0,4,1,0,1],
[4,3,1,4,4,2,1,2,0,4,2,3,0,0,1,3,0,1,2,3],
[0,1,1,3,4,1,2,4,1,3,4,0,3,0,4,2,0,2,3,2],
[0,0,2,2,0,3,3,4,2,0,1,4,4,1,3,4,2,1,3,1],
[1,4,2,2,0,4,0,0,3,3,4,2,1,0,3,4,3,1,1,2],
[1,2,0,3,1,4,4,2,0,4,3,1,1,2,4,0,0,3,3,2],
[4,3,1,2,0,3,3,1,4,4,4,2,3,0,0,2,2,1,0,1],
[2,0,2,1,4,0,1,1,0,1,3,3,2,0,2,4,3,3,4,4],
[4,3,3,3,2,2,0,3,4,1,4,0,0,0,2,1,2,1,1,4],
[3,0,4,2,1,3,4,4,3,1,0,4,1,3,2,0,2,0,1,2],
[4,1,1,0,0,0,2,2,3,3,1,3,1,4,4,0,3,2,2,4],
[2,1,0,1,2,0,4,0,4,3,1,0,1,4,3,3,2,4,3,2],
[0,4,4,4,2,4,0,3,2,3,0,1,0,3,2,3,2,1,1,1],
[4,2,2,4,0,0,0,1,3,1,3,1,1,3,0,2,2,3,4,4],
[4,0,3,2,3,2,4,0,1,3,0,1,2,4,2,1,4,3,1,0]
]
'''


'''



'''

def paired_t_test(v1, v2):

	pval_buf = []
	
	t_stat, p_val = stats.ttest_rel(v1, v2, nan_policy = 'omit')
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
	plt.rcParams["figure.figsize"] = (2.4, 3.6)

	plt.plot([start, end], [height] * 2, '-', color = color,\
			lw = linewidth, marker = TICKDOWN, markeredgewidth = linewidth, markersize = markersize)
	
	# draw the text (stars) with a bounding box that covers up the line
	box = dict(facecolor='1.', edgecolor = 'none', boxstyle = 'Square,pad=' + str(boxpad))
	plt.text(0.5 * (start + end), height, stars(p_value),\
			 ha = 'center', va = 'center', bbox = box, size = fontsize)
			 
			 
def max_height(buf):

	length = len(buf)
	max_buf = []
	for li in range(length):
		max_buf.append(np.nanmean(buf[li]))

	return max(max_buf)


def significance(ax, data, pval, barcontainer, rect_idx, ylim = 12.0, step = 1.0, useRandom = False):

	order = [[0,1]]

	hs = []
	ws = []

	rects = barcontainer.get_children()
	for rect in rects:
		hs.append(rect.get_height())
		ws.append(rect.get_width())
	
	offset = step * 1.0
	height = max_height(hs) + (offset * 2.0)
	
	for pi in range(len(pval)):

		s_rect = rects[order[pi][0]]
		e_rect = rects[order[pi][1]]

		start = s_rect.get_x() + s_rect.get_width() / 2.0
		end = e_rect.get_x() + e_rect.get_width() / 2.0
		significance_bar(start, end, height, pval[pi])

		
'''

'''

def compute_distance(u, v):

	ret = 0.0
	
	if len(u) == len(v):
	
		ret = dist.euclidean(u, v)
		
	return ret
	
	
def draw_line_ind(v1, v2, useZero = False):

	
	line1 = []
	line2 = []
	
	for i in range(len(v1)):
		line1.append(np.mean(v1[i]))
		line2.append(np.mean(v2[i]))
		
	xticks = [idx + 1 for idx in range(len(line1))]
	x_pos = [i for i, _ in enumerate(xticks)]
	

	# plot 1
	plt.subplot(2, 1, 1)
	
	legend_txt = ['Random and Bayesian', 'Random and Max-OS']
	plt.title('Distance between Random and Optimized Sequences')
	plt.xticks(x_pos, xticks)
	plt.xlabel('Index of Random Sequences')
	plt.ylabel("Distance")

	plt.plot(line1, '.-', label = legend_txt[0])
	plt.plot(line2, '.-', label = legend_txt[1])
	plt.legend()
	
	# plot 2
	plt.subplot(2, 1, 2)
	
	cross = []
	zero = []
	cnt = 0
	for j in range(len(line1)):
		diff = line1[j] - line2[j]
		if diff > 0:
			cnt += 1
		cross.append(diff)
		zero.append(0)
	
	plt.xticks(x_pos, xticks)
	plt.xlabel('Index of Random Sequences')	
	plt.ylabel('D(rand, bayesian) \n- D(rand, max-os)')
	
	dominance_txt = 'Closer to Bayesian when < 0'
	plt.plot(cross, 'r.-', label = dominance_txt)
	if useZero == True:
		plt.plot(zero, 'k')
	plt.legend()
		
	plt.show()
	
	return line1, line2, cross
	
	
def draw_bar(v1, v2):

	avg = [np.mean(v1), np.mean(v2)]
	err = [stats.sem(v1), stats.sem(v2)]
	
	fig, ax = plt.subplots()
	bar_color = 'tab:gray'
	
	xticks = ['Random\n to Bayesian', 'Random\n to Max-OS']
	x_pos = [i for i, _ in enumerate(xticks)]
	plt.xticks(x_pos, xticks)
	
	#plt.ylim(20, 50)
	
	plt.title('Distance: Random to' + '\n' + 'optimized Sequences\n')
	plt.ylabel("Distance")
	
	rects = plt.bar(x_pos, avg, color = bar_color, width = 0.7, yerr = err)
	pval = paired_t_test(v1, v2)
	significance(ax, [v1, v2], pval, rects, 0)
	
	
	plt.tight_layout()
	plt.show()
	
	
# bm_diff: diff between d_to_bayes and d_to_maxos
def draw_scatter(d_to_bayes, d_to_maxos, bm_diff):

	x = d_to_maxos
	y = d_to_bayes

	fig, ax = plt.subplots()
	
	# title
	plt.title('Similarity of Random Sequences')
	
	# label
	plt.xlabel("To Max-OS sequences")
	plt.ylabel("To Bayesian sequences")
	
	# max
	ylim = max(max(x), max(y))
	y0 = min(min(x), min(y))
	plt.ylim(y0 - 1, ylim + 1)
	plt.xlim(y0 - 1, ylim + 1)
	
	# line
	line = [i for i in range(int(y0 - 10), int(ylim + 10))]
	
	# draw
	plt.plot(line, line, '--', color = 'k', linewidth = 0.5)
	plt.scatter(y, x, color = 'b', facecolors = 'none', label = 'index of random sequences')
	
	for i in range(len(d_to_bayes)):
		ax.annotate(' %d' % i, (y[i], x[i]))
	
	plt.tight_layout()
	plt.show()
	

def plot_distance(u, v1, v2, useZero):

	v1buf = []
	v2buf = []
	
	v1ind = []
	v2ind = []
	
	if len(v1) != len(v2):
	
		print ('please check the length of all vectors')
		return
	
	for ui in u:
		
		v1temp = []
		v2temp = []
		for idx in range(len(v1)):
		
			eud = compute_distance(ui, v1[idx])	# distance between self and bayes
			v1buf.append(eud)
			v1temp.append(eud)
			
			eud = compute_distance(ui, v2[idx])	# distance between self and maxos
			v2buf.append(eud)
			v2temp.append(eud)
			
		v1ind.append(v1temp)
		v2ind.append(v2temp)
	
	tobayes, tomaxos, bm_diff = draw_line_ind(v1ind, v2ind, useZero = useZero)
	
	draw_scatter(tobayes, tomaxos, bm_diff)
	
	draw_bar(v1buf, v2buf)	

	return v1buf, v2buf
	
	
def diff_distance(u, v1, v2):

	v1ind = []
	v2ind = []
	
	if len(v1) != len(v2):
	
		print ('please check the length of all vectors')
		return
	
	for ui in u:
		
		v1temp = []
		v2temp = []
		for idx in range(len(v1)):
		
			eud = compute_distance(ui, v1[idx])	# distance between self and bayes
			v1temp.append(eud)
			
			eud = compute_distance(ui, v2[idx])	# distance between self and maxos
			v2temp.append(eud)
			
		v1ind.append(v1temp)
		v2ind.append(v2temp)
		
	line1 = []
	line2 = []
	diff = []
	
	for i in range(len(v1ind)):
		line1.append(np.mean(v1ind[i]))
		line2.append(np.mean(v2ind[i]))
		diff.append(line1[i] - line2[i])

	return line1, line2, diff	# distance to bayes, maxos and their difference
	
	
def diff_distance_rand_to_bayes_maxos():

	u = seq_random_5
	v1 = seq_bayes_5
	v2 = seq_maxos_5
	
	r2bayes, r2maxos, diff = diff_distance(u, v1, v2)
	
	return r2bayes, r2maxos, diff
	
	
def diff_distance_rand_to_maxos_minos():

	u = seq_random_5
	v1 = seq_maxos_5
	v2 = seq_minos_5
	
	r2bayes, r2maxos, diff = diff_distance(u, v1, v2)
	
	return r2bayes, r2maxos, diff
	

def diff_distance_rand_to_bayes_minos():

	u = seq_random_5
	v1 = seq_maxos_5
	v2 = seq_minos_5
	
	r2bayes, r2maxos, diff = diff_distance(u, v1, v2)
	
	return r2bayes, r2maxos, diff
	

def main():

	plt.rcParams["figure.figsize"] = (2.4, 3.6)

	# plot the score: opt vs counter-opt sequences
	#v1, v2 = plot_distance(seq_random, seq_bayes, seq_maxos, False)
	v1, v2 = plot_distance(seq_random_5, seq_bayes_5, seq_maxos_5, True)
	v1, v2 = plot_distance(seq_random_5, seq_maxos_5, seq_minos_5, True)
	v1, v2 = plot_distance(seq_random_5, seq_bayes_5, seq_minos_5, True)
	
if __name__ == '__main__':
	main()


	
