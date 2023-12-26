import math
import scipy.stats as stats
import scipy.spatial.distance as dist
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.markers import TICKDOWN

#
# original rmse
#
rmse = [[0.08903265834278944, 0.09335792330699881, 0.006635560954832899], [0.09665181909998195, 0.05272272730196348, 0.008300265969005027], [0.11708781471836426, 0.07990025960709224, 0.006217366351706096], [0.10577590791427974, 0.04687814601199437, 0.008721050665085682], [0.1474458526038242, 0.08577173407798291, 0.007086769968100792], [0.08560584824357921, 0.07637395574777617, 0.008558914729065727], [0.10602093051537742, 0.06888165587578951, 0.007285825127495359], [0.10720494605464861, 0.06572336509608684, 0.007034058168683053], [0.11159981558880311, 0.11687341811866889, 0.007026911442780323], [0.10760966327957275, 0.07597726687413392, 0.00805640933619525], [0.10772942176274095, 0.08228194092795055, 0.00621489628583486], [0.1237028270828244, 0.06950664427152567, 0.005352197756076735], [0.0973514213272012, 0.0510284534623209, 0.006037340119501143], [0.1189652586112398, 0.10283871763772243, 0.007657846777949782], [0.10926013194829587, 0.058249970807701945, 0.008134391086038945], [0.10146867468211127, 0.055109804683389066, 0.005969870729867672], [0.11607727287799352, 0.07130777698245006, 0.007114535267520692], [0.11009102989806849, 0.0773248526106067, 0.007012707984285948], [0.12006352451684504, 0.08725549794575466, 0.005656366581784849], [0.10398210070245195, 0.07787260006808054, 0.006269282550789493], [0.10525436029698249, 0.05628114070803028, 0.006902141032194052], [0.10330436333088351, 0.055364243350742866, 0.006530712146462836], [0.10730264642574261, 0.06031919811918637, 0.007808833179971634], [0.09566783414546089, 0.04832793474217652, 0.0060981002510815], [0.10153105529836458, 0.051008409515745395, 0.007810360067617713], [0.09074482584283775, 0.10512576942516001, 0.008808779391517606], [0.09669525614617797, 0.06292058043463736, 0.006582283193431682], [0.1034000794699296, 0.054438332862433235, 0.006377272918175171], [0.09535102182801466, 0.05581878622730803, 0.006545412365632124], [0.11007987726691028, 0.07132347254795635, 0.005518018836434714], [0.10719464259315105, 0.08375352646833567, 0.005887956014615883], [0.10141121918100506, 0.08345401812639602, 0.008355479313669928], [0.10818691970608044, 0.05296646848014323, 0.006191839738895517], [0.09443975366948881, 0.048989092339148596, 0.007896418123602035], [0.10102895291963992, 0.04458730158088427, 0.0067775845887658465], [0.12449304795898346, 0.10189581368508384, 0.006084609176415572], [0.0994150198745882, 0.06602428697669963, 0.006157300086382236], [0.11882074267156913, 0.06461319780408938, 0.006938136265495371], [0.11225007091492342, 0.10958685565793379, 0.007737787231618593], [0.10015409059068527, 0.06681306134970072, 0.008573420560335401], [0.11868977305700698, 0.08859411848182608, 0.007584356050954489], [0.11587957982794823, 0.07147789100199972, 0.006526522660401179], [0.10066876007722626, 0.054688554536342034, 0.007312104799656088], [0.1076975835308754, 0.058406513513874415, 0.006243909790969428], [0.11144493127643967, 0.058976784359839184, 0.007858971288781302], [0.09563592133243477, 0.049339029181438995, 0.0072250079540136835], [0.10580416407213245, 0.05870943260824553, 0.007083180016726432], [0.10886378331353593, 0.05459624651412098, 0.007213323785452536], [0.09773171164275749, 0.0520331551242592, 0.004878487112385133], [0.09500744036874006, 0.047938106102245956, 0.006265457417343031], [0.10598596366096838, 0.0628000081705889, 0.00797929227805739], [0.1089020942078143, 0.0606180791449813, 0.008710985301922282], [0.09210502226351454, 0.09045383393584108, 0.0071363641835589745], [0.11007353089167532, 0.057851199718024014, 0.007894112927446753], [0.10547705359677972, 0.06686664846889356, 0.008873125205230361], [0.09500771234334167, 0.04640150315766624, 0.008506256319268234], [0.1020794773577845, 0.05153221413900379, 0.006504211587557381], [0.10459960298698184, 0.053007315063187524, 0.0073865249042784], [0.09238445432530056, 0.05670176207361223, 0.006623083881029361], [0.09328914047725764, 0.06123664743785892, 0.008634235414957666], [0.09282171410344339, 0.054506715832444985, 0.007808469168452871], [0.10612130006462746, 0.06029714721693198, 0.007319963875574623], [0.12242116197907975, 0.07010911550602461, 0.007665847044603485], [0.09681445657382329, 0.05350569412228536, 0.00768579595173134], [0.1107461175009174, 0.0558650211993824, 0.005204892569393696], [0.10705372838862513, 0.04964873776499448, 0.009457861152200365], [0.11093178737293875, 0.07017556744259372, 0.007859562998620622], [0.12443372124232524, 0.08543004285893563, 0.006072590632825149], [0.11518851218964637, 0.07213772774500882, 0.00773791521115271], [0.09714258730173501, 0.0896578840535776, 0.005469149892102265], [0.09460203658077289, 0.05722751061486871, 0.0069307475754580225], [0.11190444538627872, 0.07404949267366943, 0.008586256554527931]]

#
# normalised rmse
#
#rmse = [[0.05541413094422767, 0.6746886409456192, 0.3836930172488509], [0.178621767101683, 0.11254478882990968, 0.7472154112918794], [0.5090873909288373, 0.4885164637077932, 0.2923716708179146], [0.32616523687807547, 0.0316913473960473, 0.8391023575037754], [1.0, 0.569741943123631, 0.4822237354965567], [0.0, 0.43973387545693815, 0.8036966591244186], [0.33012743907441316, 0.336086035030065, 0.5256915015414109], [0.34927387270617316, 0.29239450848289184, 0.470713035789689], [0.4203422624907609, 1.0, 0.4691524019911408], [0.3558184586762267, 0.4342461152528761, 0.6939643270411557], [0.3577550446193741, 0.5214644409257059, 0.29183228140578976], [0.6160571822943879, 0.3447320714430112, 0.10344440955749372], [0.18993486829656267, 0.08910634835487091, 0.2530592602919955], [0.5394470895138926, 0.8058451449164465, 0.6069300392148771], [0.382507794904414, 0.18900820629471887, 0.7109932373607072], [0.25651399288596705, 0.14556741469165219, 0.23832593887145628], [0.49274615921604686, 0.36964878847238597, 0.4882868566084152], [0.395944047996058, 0.45288850193813107, 0.4660507862744765], [0.5572068862177765, 0.5902682065174621, 0.1698658949097553], [0.2971580071667372, 0.46046599376794217, 0.3037086349165039], [0.31773141442458713, 0.16177157782481696, 0.4419062304617006], [0.28619847735137166, 0.1490872976171766, 0.3607971350914953], [0.35085376216615477, 0.21763372127037542, 0.6399010087642315], [0.16270998047261157, 0.05174760162052677, 0.26632747796805334], [0.257522734992289, 0.0888290620994248, 0.6402344359166772], [0.0831011842968469, 0.8374840252018758, 0.8582597195513323], [0.17932417724290767, 0.2536210233976274, 0.37205872816523483], [0.28774628026691645, 0.13627833052007643, 0.32729054075052383], [0.15758688385055034, 0.1553754051865957, 0.3640072286635594], [0.39576370145058815, 0.36986591959324333, 0.13965483458856856], [0.34910725788128505, 0.5418222303722576, 0.22043818509996174], [0.25558489364510173, 0.5376788573943622, 0.7592723745765664], [0.36515313503143754, 0.11591668359828264, 0.2867974127231098], [0.14285098323163517, 0.06089399969305963, 0.6590269728957693], [0.2494033568661219, 0.0, 0.4147067830382639], [0.6288356561048962, 0.7928010916763517, 0.26338142583327895], [0.22330482951722552, 0.29655743623479963, 0.27925497303310476], [0.5371101566309519, 0.2770365484046634, 0.4497665259930023], [0.4308573866866179, 0.899198313455858, 0.624386672583056], [0.23525616625697823, 0.30746927395385587, 0.8068643041220872], [0.5349922781489389, 0.608786569381398, 0.5908818356053162], [0.489549311931017, 0.37200213137829163, 0.35988227510730714], [0.24357876409417797, 0.13973987591625217, 0.531430205550352], [0.3572401961455597, 0.19117380480339796, 0.2981679737694863], [0.41783766511943526, 0.1990628832776401, 0.650849690477877], [0.16219392596459103, 0.0657349962640607, 0.512410827599317], [0.32662216048513254, 0.1953643618409002, 0.4814397961757791], [0.3760985354151837, 0.13846289457263808, 0.5098593503756703], [0.19608445252590592, 0.10300530585956869, 0.0], [0.1520309098038299, 0.04635474530728446, 0.302873338779258], [0.3295619983896849, 0.2519530369318521, 0.677124239844223], [0.376718051773158, 0.22176841600997585, 0.8369043795539755], [0.10509659705188255, 0.6345137151057485, 0.49305364697069864], [0.39566107572634035, 0.18349164089077305, 0.6585235861587961], [0.32133253480129265, 0.3082105936118959, 0.8723109442718515], [0.15203530784041508, 0.025097510610266147, 0.7921976181333014], [0.26639113765644695, 0.09607533079314547, 0.3550101959432523], [0.30714348971833394, 0.1164817517607553, 0.5476813577766757], [0.10961522645168342, 0.1675904180908048, 0.3809683929453857], [0.12424469100810383, 0.2303256372649634, 0.8201444716937928], [0.11668605030860957, 0.1372243347223633, 0.6398215194026728], [0.33175049117941185, 0.2173286709604331, 0.5331463955471081], [0.5953316807844211, 0.3530666073588266, 0.6086770610969389], [0.18125173900294453, 0.12337628535819004, 0.6130333130550472], [0.40653731378939034, 0.15601501586550384, 0.07127730868250562], [0.34682856780058857, 0.0700194785185969, 1.0], [0.40953973712267067, 0.353985897808388, 0.650978902425662], [0.6278762978824636, 0.5650150158046257, 0.26075693098181507], [0.47837422154331116, 0.3811302568692243, 0.6244146195323561], [0.18655786294821847, 0.623502612000667, 0.12898330090130913], [0.14547522158612664, 0.17486357878109673, 0.4481530543758976], [0.4252683584803571, 0.407577450607476, 0.8096673060347782]]

xl = ['rmse against oneshot+', 'rmse against oneshot-', 'rmse against bayesian']
yl = ['confidence', 'score']


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
				linewidth = 1.0, markersize = 3, boxpad = 0.3, fontsize = 10, color = 'k'):

	fontsize = 10
	
	t_stat, p_val = stats.ttest_rel(data1, data2)
	star = stars(p_val)
	
	plt.plot([start, end], [height] * 2, '-', color = color,\
			lw = linewidth, marker = TICKDOWN, markeredgewidth = linewidth, markersize = markersize)
	
	box = dict(facecolor='1.', edgecolor = 'none', boxstyle = 'Square,pad=' + str(boxpad))
	plt.text(0.5 * (start + end), height, star, ha = 'center', va = 'center', bbox = box, size = fontsize)
	
	
def rmse_under_percentile(pct_value, seq_category):	# seq_category - 0:oneshot+, 1:oneshot-, 2:bayesian

	t_rmse = np.array(rmse).T.tolist()
	
	u_buf = t_rmse[seq_category]
	
	val = np.percentile(u_buf, pct_value)
	
	idx_under = []
	
	for i in range(len(osp)):
	
		if item < val:
			idx_under.append(i)
	
	return idx_under
	

def rmse_percentile_values(pct_value):

	t_rmse = np.array(rmse).T.tolist()
	
	upper = []
	lower = []
	
	pct_upper = 100 - pct_value
	pct_lower = pct_value

	for item in t_rmse:
	
		u = np.percentile(item, pct_upper)
		l = np.percentile(item, pct_lower)
		
		upper.append(u)
		lower.append(l)
	
	return upper, lower
	

def rmse_percentile(pct_value):

	t_rmse = np.array(rmse).T.tolist()
	
	pctbuf = []
	
	for item in t_rmse:
	
		b = np.percentile(item, pct_value)
		pctbuf.append(b)
	
	return pctbuf 
	

def __validate__(score, visit_cnt):

	res = 0.0

	if visit_cnt > 0.0:

		res = score / visit_cnt

	return res	
	
	
def draw_scatter_regression(X, Y):

	for i in range(len(X)):
	
		for j in range(len(Y)):
		
			fig, ax = plt.subplots()
			
			# label
			plt.xlabel(xl[i])
			plt.ylabel(yl[j])
			
			#regression part
			slope, intercept, r_value, p_value, std_err = stats.linregress(X[i],Y[j])
			print ('slope = %f, intercept = %f, r_sqaured = %f, p_value = %f\n' % (slope, intercept, r_value**2, p_value))
			line = slope * np.array(X[i]) + intercept
			plt.plot(X[i], line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))

			plt.scatter(X[i], Y[j], color = 'b', facecolors = 'none')
			plt.tight_layout()
			
	plt.show()
	
	
def draw_bar_sem(uY, lY, index):

	plt_metric = {'figx':2.4, 'figy':3.6}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])

	# index = 0:confidence, 1: score
	i = index
	
	ind = np.arange(2)
	width = 0.8
	xticks = ['Low', 'High']
	ylabel = ['Causal rating', 'Test score']
	
	ylim = [5, 8]
	offset = 0.4
	if index == 1:
		ylim = [6.0, 9.0]
		offset = 0.125
	ytop = ylim[1]
	
	ubuf = []
	lbuf = []
	
	for u, l in zip(uY[i], lY[i]):
	
		ubuf.extend(u)
		lbuf.extend(l)
	
	
	avg = []
	err = []
	
	avg.append(np.mean(lbuf))
	avg.append(np.mean(ubuf))
		
	err.append(stats.sem(lbuf))
	err.append(stats.sem(ubuf))
	
	plt.xticks(ind, xticks)
	plt.xlabel('Subjective experience\n(RMSE)')
	plt.ylabel(ylabel[i])
	plt.ylim(ylim[0], ylim[1])

	rects = plt.bar(ind, avg, width, color = 'gray', yerr = err)
	t_stat, p_val = stats.ttest_rel(lbuf, ubuf)
	print (p_val)
	
	significance(ubuf, lbuf, rects, 0.0, 1.0, ytop - offset)
	

	plt.tight_layout()
	plt.show()
	


def rmse_draw_plot_ext(conf_map, efficiency = False):

	# 0: index, 1: confidence, 2: score
	
	X = [[], [], []]	# rmse against 0: oneshot+, 1: oneshot-, 2: bayesian
	Y = [[], []] 	# performance 0: confidence, 1: score
	
	upper_Y = [[], []]
	lower_Y = [[], []]
	
	upper, lower = rmse_percentile_values(50)
	i = 0
	
	indices = []
	
	for cm in conf_map:
		
		rmse_index = cm[0]
		rmse_vals = rmse[rmse_index]
		
		if (rmse_vals[i] > upper[i]) or (rmse_vals[i] < lower[i]):
	
			X[i].append(rmse_vals[i])
			
			c = cm[1]
			s = cm[2]
			
			if efficiency == False:
				denom = 1.0
			else:
				denom = 4.0
			
			cbuf = [np.mean([c[0]/denom, c[2]/denom]), np.mean([c[1]/denom, c[3]/denom]), c[4]/denom]
			sbuf = [s[0]/denom + s[2]/denom, s[1]/denom + s[3]/denom, s[4]/denom]

			Y[0].append(np.mean(cbuf))
			Y[1].append(np.mean(sbuf))
			
			if (rmse_vals[i] > upper[i]):
				
				indices.append(rmse_index)
				upper_Y[0].append(cbuf)
				upper_Y[1].append(sbuf)
				
			if (rmse_vals[i] < lower[i]):
			
				lower_Y[0].append(cbuf)
				lower_Y[1].append(sbuf)

	#draw_scatter_regression([X[i]], Y)
	
	#print (indices)
	
	draw_bar_sem(upper_Y, lower_Y, 0)
	draw_bar_sem(upper_Y, lower_Y, 1)
	

def draw_bar_sem_25(y25, y50, y75, y100, index, yrange = [0, 10], offset = 0.5):

	plt_metric = {'figx':4.0, 'figy':3.6}
	plt.rcParams["figure.figsize"] = (plt_metric['figx'], plt_metric['figy'])

	# index = 0:confidence, 1: score
	i = index
	ylim = yrange
	
	ind = np.arange(4)
	width = 0.8
	xticks = ['(0, 25]', '(25, 50]', '(50, 75]','(75, 100]']
	ylabel = ['Causal rating', 'Test score', 'efficiency']
	ytop = ylim[1]
	denom = 1.0
	
	b25 = []
	b50 = []
	b75 = []
	b100 = []
	
	if index >= 2:
		denom = 4.0
	
	for r25, r50, r75, r100 in zip(y25[i], y50[i], y75[i], y100[i]):
	
		b25.extend(r25)
		b50.extend(r50)
		b75.extend(r75)
		b100.extend(r100)

	
	avg = []
	err = []
	
	avg.append(np.mean(b100))
	avg.append(np.mean(b75))
	avg.append(np.mean(b50))
	avg.append(np.mean(b25))
		
	err.append(stats.sem(b100))
	err.append(stats.sem(b75))
	err.append(stats.sem(b50))
	err.append(stats.sem(b25))
	
	plt.xticks(ind, xticks)
	plt.xlabel('Subjective experience\n(RMSE)')
	plt.ylabel(ylabel[i])
	plt.ylim(ylim[0], ylim[1])


	rects = plt.bar(ind, avg, width, color = 'gray', yerr = err)
	
	t_stat, p_val = stats.ttest_rel(b100, b75)
	print (p_val)
	significance(b75, b100, rects, 0.0, 1.0, ytop - (offset * 3))
	
	t_stat, p_val = stats.ttest_rel(b100, b50)
	print (p_val)
	significance(b50, b100, rects, 0.0, 2.0, ytop - (offset * 2))
	
	t_stat, p_val = stats.ttest_rel(b100, b25)
	print (p_val)
	significance(b25, b100, rects, 0.0, 3.0, ytop - (offset * 1))
	
	plt.tight_layout()
	plt.show()	
	
	
def rmse_draw_plot_25(conf_map, efficiency = False):

	# 0: index, 1: confidence, 2: score
	
	X = [[], [], []]	# rmse against 0: oneshot+, 1: oneshot-, 2: bayesian
	Y = [[], []] 		# performance 0: confidence, 1: score
	
	y25 = [[], []]
	y50 = [[], []]
	y75 = [[], []]
	y100 = [[], []]
	
	v25, v75 = rmse_percentile_values(25)
	v50 = rmse_percentile(50)

	i = 0
	yrange = []
	
	for cm in conf_map:
		
		rmse_index = cm[0]
		rmse_vals = rmse[rmse_index]
		
		#if (rmse_vals[i] > upper[i]) or (rmse_vals[i] < lower[i]):
	
		X[i].append(rmse_vals[i])
		
		c = cm[1]
		s = cm[2]
		
		if efficiency == False:
			denom = 1.0
		else:
			denom = 4.0
		
		'''
		cbuf = [np.mean([c[0]/denom, c[2]/denom]), np.mean([c[1]/denom, c[3]/denom]), c[4]/denom]
		sbuf = [np.mean([s[0]/denom, s[2]/denom]), np.mean([s[1]/denom, s[3]/denom]), s[4]/denom]
		'''
		cbuf = [np.mean([c[0]/denom, c[2]/denom]), np.mean([c[1]/denom, c[3]/denom]), c[4]/denom]
		sbuf = [s[0]/denom + s[2]/denom, s[1]/denom + s[3]/denom, s[4]/denom]

		Y[0].append(np.mean(cbuf))
		Y[1].append(np.mean(sbuf))
		
		if (rmse_vals[i] > v25[i]):
			
			y25[0].append(cbuf)
			y25[1].append(sbuf)
			
		elif (rmse_vals[i] > v50[i]):
		
			y50[0].append(cbuf)
			y50[1].append(sbuf)
			
		elif (rmse_vals[i] > v75[i]):
		
			y75[0].append(cbuf)
			y75[1].append(sbuf)
			
		else:
			y100[0].append(cbuf)
			y100[1].append(sbuf)

	#draw_scatter_regression([X[i]], Y)
	
	idx = 0
	if efficiency == False:
		yrange = [5, 9]
		offset = 0.35
	else:
		yrange = [1.4, 2.1]
		offset = 0.1
		
	draw_bar_sem_25(y25, y50, y75, y100, 0, yrange = yrange, offset = offset)

	if efficiency == False:
		yrange = [6, 10]
		offset = 0.25
	else:
		yrange = [1.6, 2.4]
		offset = 0.08
		
	draw_bar_sem_25(y25, y50, y75, y100, 1, yrange = yrange, offset = offset)

	
def rmse_draw_plot(conf_map, efficiency = False):

	# 0: index, 1: confidence, 2: score
	
	X = [[], [], []]	# rmse against 0: oneshot+, 1: oneshot-, 2: bayesian
	Y = [[], []] 	# performance 0: confidence, 1: score
	
	for cm in conf_map:
		
		rmse_index = cm[0]
		rmse_vals = rmse[rmse_index]
		
		X[0].append(rmse_vals[0])
		X[1].append(rmse_vals[1])
		X[2].append(rmse_vals[2])

		c = cm[1]
		s = cm[2]
		
		if efficiency == False:
			cbuf = [np.mean([c[0], c[2]]), np.mean([c[1], c[3]]), c[4]]
			sbuf = [np.mean([s[0], s[2]]), np.mean([s[1], s[3]]), s[4]]
		else:
			visit_cnt = [4, 4, 4, 4, 4]
			
			c1 = np.mean([__validate__(c[0], visit_cnt[0]), __validate__(c[2], visit_cnt[2])])
			c2 = np.mean([__validate__(c[1], visit_cnt[1]), __validate__(c[3], visit_cnt[3])])
			c3 = __validate__(c[4], visit_cnt[4])
			
			s1 = np.mean([__validate__(s[0], visit_cnt[0]), __validate__(s[2], visit_cnt[2])])
			s2 = np.mean([__validate__(s[1], visit_cnt[1]), __validate__(s[3], visit_cnt[3])])
			s3 = __validate__(s[4], visit_cnt[4])
			
			cbuf = [c1, c2, c3]
			sbuf = [s1, s2, s3]

		Y[0].append(np.mean(cbuf))
		Y[1].append(np.mean(sbuf))

	draw_scatter_regression(X, Y)

	