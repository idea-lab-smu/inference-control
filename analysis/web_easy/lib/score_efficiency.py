import numpy as np


'''
sequences for 17 nodes

	bayes_seq_visits = [ [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
						[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
						[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] ]

	#					  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
	maxos_seq_visits = [ [1, 2, 2, 1, 1, 1, 2, 3, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2],
						[3, 3, 1, 1, 4, 1, 1, 3, 4, 1, 2, 3, 1, 3, 1, 2, 1, 1],
						[2, 2, 5, 2, 1, 1, 4, 1, 1, 2, 1, 2, 4, 1, 1, 4, 2, 1] ]

	minos_seq_visits = [ [0, 3, 0, 0, 4, 0, 3, 1, 3, 2, 3, 2, 2, 0, 3, 8, 2, 0],
						[3, 0, 1, 0, 2, 7, 2, 3, 3, 2, 1, 3, 3, 0, 2, 2, 1, 1],
						[1, 2, 2, 1, 1, 0, 1, 3, 1, 3, 3, 1, 3, 1, 1, 9, 3, 0] ]

	rand_seq_visits = [ [3, 3, 0, 0, 1, 3, 2, 3, 1, 2, 4, 4, 3, 2, 1, 0, 3, 1],
						[2, 0, 6, 2, 1, 3, 1, 1, 3, 3, 0, 3, 1, 3, 1, 2, 3, 1],
						[5, 2, 3, 6, 1, 2, 2, 1, 3, 1, 1, 1, 3, 2, 0, 0, 0, 3] ]
'''

def number_of_visits(visits, nodes):

	ret = []

	for node in nodes:

		ret.append(visits[node])

	return ret



#ontology용
# // For random var
# _used_perms = {};
# List
# used_perms = [];
#
# // len 60(0~19) var
# seed_list = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11,
# 			 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19];
#

def visits_on_node_20(idx, efficiency):
	# bayes_seq_visits = [[4, 3, 2, 2, 2, 2, 3, 2, 4, 3, 3, 3, 6, 4, 2, 2, 3, 7, 2, 3],
	# 					[2, 3, 4, 4, 2, 2, 3, 4, 4, 2, 4, 3, 3, 2, 2, 3, 2, 3, 6, 4],
	# 					[3, 2, 2, 3, 2, 2, 4, 5, 2, 4, 3, 3, 4, 3, 3, 3, 6, 4, 2, 2],
	# 					[3, 4, 2, 2, 2, 3, 2, 3, 3, 8, 3, 3, 4, 3, 2, 2, 3, 2, 3, 5]]
	#
	# maxos_seq_visits = [[4, 3, 2, 3, 5, 4, 5, 4, 3, 3, 2, 3, 2, 2, 6, 2, 3, 5, 3, 3],
	# 					[3, 5, 4, 4, 3, 2, 2, 3, 3, 5, 2, 2, 3, 3, 4, 2, 4, 7, 2, 3],
	# 					[5, 2, 2, 4, 4, 4, 5, 4, 2, 3, 3, 4, 2, 2, 4, 3, 3, 3, 2, 4],
	# 				 	[2, 3, 3, 4, 4, 3, 2, 3, 3, 2, 3, 3, 3, 9, 4, 3, 2, 7, 2, 2]]
	#
	# minos_seq_visits = [[2, 3, 1, 0, 2, 6, 2, 4, 3, 4, 3, 3, 1, 4, 4, 4, 3, 2, 5, 4],
	# 					[4, 4, 2, 3, 7, 1, 1, 4, 6, 0, 3, 3, 2, 3, 3, 1, 4, 4, 3, 2],
	# 					[4, 3, 5, 4, 0, 4, 6, 2, 0, 1, 2, 6, 2, 4, 4, 3, 2, 2, 3, 3],
	# 					[6, 1, 2, 2, 7, 4, 3, 3, 5, 3, 2, 2, 3, 3, 1, 0, 3, 5, 2, 3]]

	maxos_seq_visits = [[4, 3, 2, 3, 5, 4, 5, 4, 3, 3, 2, 3, 2, 2, 6, 2, 3, 5, 3, 3],
						[3, 5, 4, 4, 3, 2, 2, 3, 3, 5, 2, 2, 3, 3, 4, 2, 4, 7, 2, 3],
						[5, 2, 2, 4, 4, 4, 5, 4, 2, 3, 3, 4, 2, 2, 4, 3, 3, 3, 2, 4],
					 	[2, 3, 3, 4, 4, 3, 2, 3, 3, 2, 3, 3, 3, 9, 4, 3, 2, 7, 2, 2]]

	minos_seq_visits = [[2, 3, 1, 0, 2, 6, 2, 4, 3, 4, 3, 3, 1, 4, 4, 4, 3, 2, 5, 4],
						[4, 4, 2, 3, 7, 1, 1, 4, 6, 0, 3, 3, 2, 3, 3, 1, 4, 4, 3, 2],
						[4, 3, 5, 4, 0, 4, 6, 2, 0, 1, 2, 6, 2, 4, 4, 3, 2, 2, 3, 3],
						[6, 1, 2, 2, 7, 4, 3, 3, 5, 3, 2, 2, 3, 3, 1, 0, 3, 5, 2, 3]]

	bayes_seq_visits = [[4, 3, 2, 2, 2, 2, 3, 2, 4, 3, 3, 3, 6, 4, 2, 2, 3, 7, 2, 3],
						[2, 3, 4, 4, 2, 2, 3, 4, 4, 2, 4, 3, 3, 2, 2, 3, 2, 3, 6, 4],
						[3, 2, 2, 3, 2, 2, 4, 5, 2, 4, 3, 3, 4, 3, 3, 3, 6, 4, 2, 2],
						[3, 4, 2, 2, 2, 3, 2, 3, 3, 8, 3, 3, 4, 3, 2, 2, 3, 2, 3, 5]]


	random_seq_visits = [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
						 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
						 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
						 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]

	nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

	visits = []
	def_buf = [1.0 for i in range(len(nodes))]

	'''
    bayes_visit = number_of_visits(bayes_seq_visits, nodes) if efficiency == True else def_buf
    visits.append(bayes_visit)

    maxos_visit = number_of_visits(maxos_seq_visits, nodes) if efficiency == True else def_buf
    visits.append(maxos_visit)

    minos_visit = number_of_visits(minos_seq_visits, nodes) if efficiency == True else def_buf
    visits.append(minos_visit)

    random_visit = number_of_visits(random_seq_visits, nodes) if efficiency == True else def_buf
    visits.append(random_visit)
    '''
	visits.extend(maxos_seq_visits)
	visits.extend(minos_seq_visits)
	visits.extend(bayes_seq_visits)
	visits.extend(random_seq_visits)
	print("visits[idx]:", visits[idx])
	ret = number_of_visits(visits[idx], nodes) if efficiency == True else def_buf

	return ret  # bayes_visit, maxos_visit, minos_visit, rand_visit


def visits_on_node_5(idx, efficiency):

	bayes_seq_visits = [ [4, 3, 4, 5, 4],
						[4, 3, 4, 4, 5], 
						[5, 4, 4, 3, 4 ]]

	#					  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
	maxos_seq_visits = [ [5, 4, 5, 4, 2],
						[4, 4, 6, 4, 2],
						[5, 4, 5, 4, 2] ]

	minos_seq_visits = [ [6, 1, 3, 10, 0],
						[5, 4, 4, 6, 1],
						[4, 3, 5, 7, 1] ]
						
	random_seq_visits = [[4, 4, 4, 4, 4],
						[4, 4, 4, 4, 4],
						[4, 4, 4, 4, 4]]

	
	nodes = [0, 1, 2, 3, 4]

	visits = []
	def_buf = [1.0 for i in range(len(nodes))]

	'''
	bayes_visit = number_of_visits(bayes_seq_visits, nodes) if efficiency == True else def_buf
	visits.append(bayes_visit)

	maxos_visit = number_of_visits(maxos_seq_visits, nodes) if efficiency == True else def_buf
	visits.append(maxos_visit)

	minos_visit = number_of_visits(minos_seq_visits, nodes) if efficiency == True else def_buf
	visits.append(minos_visit)
	
	random_visit = number_of_visits(random_seq_visits, nodes) if efficiency == True else def_buf
	visits.append(random_visit)
	'''
	
	visits.extend(bayes_seq_visits)
	visits.extend(maxos_seq_visits)
	visits.extend(minos_seq_visits)
	visits.extend(random_seq_visits)
	print("idx: ", idx)
	print("visits[idx]:", visits)
	ret = number_of_visits(visits[idx], nodes) if efficiency == True else def_buf
	
	return ret #bayes_visit, maxos_visit, minos_visit, rand_visit


def visits_on_node_17(idx, efficiency):

	#			0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
	visits = [  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], # bayes 1
				[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], # bayes 2
				[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], # bayes 3
				[1, 2, 2, 1, 1, 1, 2, 3, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2], # maxos 1
				[3, 3, 1, 1, 4, 1, 1, 3, 4, 1, 2, 3, 1, 3, 1, 2, 1, 1], # maxos 2
				[2, 2, 5, 2, 1, 1, 4, 1, 1, 2, 1, 2, 4, 1, 1, 4, 2, 1], # maxos 3
				[0, 3, 0, 0, 4, 0, 3, 1, 3, 2, 3, 2, 2, 0, 3, 8, 2, 0], # minos 1
				[3, 0, 1, 0, 2, 7, 2, 3, 3, 2, 1, 3, 3, 0, 2, 2, 1, 1], # minos 2
				[1, 2, 2, 1, 1, 0, 1, 3, 1, 3, 3, 1, 3, 1, 1, 9, 3, 0], # minos 3
				[3, 3, 0, 0, 1, 3, 2, 3, 1, 2, 4, 4, 3, 2, 1, 0, 3, 1], # random 1
				[2, 0, 6, 2, 1, 3, 1, 1, 3, 3, 0, 3, 1, 3, 1, 2, 3, 1], # random 2
				[5, 2, 3, 6, 1, 2, 2, 1, 3, 1, 1, 1, 3, 2, 0, 0, 0, 3] ]  # random 3
	
	# 7->12, 8->12, 9->13, 10->13, 11->14, 14->16
	nodes = [8, 9, 7, 10, 11, 17]

	def_buf = [1.0 for i in range(len(nodes))]

	ret = number_of_visits(visits[idx], nodes) if efficiency == True else def_buf

	return ret #bayes_visit, maxos_visit, minos_visit, rand_visit


def count_visits_on_each_node(sequences):

	'''
	history = [
	[4, 15, 15, 4, 15, 8, 15, 8, 16, 1, 6, 12, 16, 4, 15, 15, 10, 9, 14, 6, 15, 6, 11, 12, 14, 8, 10, 7, 9, 4, 1, 10, 14, 11, 1, 15],
	[8, 9, 6, 14, 5, 0, 10, 5, 16, 5, 12, 7, 15, 5, 4, 9, 15, 8, 17, 7, 8, 5, 5, 2, 11, 5, 6, 11, 14, 12, 7, 11, 4, 0, 12, 0],
	[7, 15, 11, 16, 2, 12, 15, 15, 15, 12, 15, 15, 15, 15, 0, 8, 1, 16, 10, 9, 6, 2, 9, 7, 15, 3, 4, 7, 9, 16, 10, 1, 13, 14, 12, 10],
	[5, 14, 0, 11, 10, 16, 12, 0, 11, 1, 8, 7, 16, 6, 0, 13, 12, 1, 17, 11, 1, 7, 10, 7, 5, 11, 10, 12, 16, 5, 13, 4, 10, 6, 9, 9],
	[3, 15, 13, 4, 5, 5, 8, 0, 2, 16, 13, 9, 11, 11, 7, 8, 2, 9, 17, 9, 6, 0, 2, 13, 14, 2, 8, 11, 2, 5, 2, 3, 16, 16, 15, 12],
	[2, 11, 0, 0, 2, 5, 9, 10, 3, 8, 13, 1, 0, 2, 17, 3, 3, 0, 8, 4, 13, 3, 0, 5, 17, 12, 6, 1, 6, 17, 3, 7, 8, 12, 12, 3]

	]
	'''

	ret = []

	for seq in sequences:
	
		visits = []

		for i in range(len(seq)):

			visits.append(seq.count(i))

		print (visits)

		ret.append(visits)

	return ret
