import math

'''
  Purpose:

	ALNORM computes the cumulative density of the standard normal distribution.

  Licensing:

	This code is distributed under the GNU LGPL license. 

  Modified:

	01 November 2010

  Author:

	Original FORTRAN77 version by David Hill.
	C++ version by John Burkardt.
	Python version by JeeHang Lee (jeehanglee@gmail.com)

  Reference:

	David Hill,
	Algorithm AS 66:
	The Normal Integral,
	Applied Statistics,
	Volume 22, Number 3, 1973, pages 424-427.

  Parameters:

	Input, X, is one endpoint of the semi-infinite interval
	over which the integration takes place.

	Input, int UPPER, determines whether the upper or lower
	interval is to be integrated:
	1  => integrate from X to + Infinity
	0 => integrate from - Infinity to X.

	Output, ALNORM, the integral of the standard normal
	distribution over the desired interval.
'''
def alnorm(x, upper):
	#print '__alnorm__'
	a1 = 5.75885480458
	a2 = 2.62433121679
	a3 = 5.92885724438
	b1 = -29.8213557807
	b2 = 48.6959930692
	c1 = -0.000000038052
	c2 = 0.000398064794
	c3 = -0.151679116635
	c4 = 4.8385912808
	c5 = 0.742380924027
	c6 = 3.99019417011
	con = 1.28
	d1 = 1.00000615302
	d2 = 1.98615381364
	d3 = 5.29330324926
	d4 = -15.1508972451
	d5 = 30.789933034
	ltone = 7.0
	p = 0.398942280444
	q = 0.39990348504
	r = 0.398942280385
	up = 0
	utzero = 18.66
	value = 0.0
	y = 0.0
	z = 0.0
	
	up = upper
	z = x

	if z < 0.0:
		up = not up
		z = -z

	if (ltone < z) and ((not up) or (utzero < z)):
		if up:
			value = 0.0
		else:
			value = 1.0
		return value

	y = 0.5 * z * z
	
	if z <= con:
		value = 0.5 - z * ( p - q * y 
				/ ( y + a1 + b1 
				/ ( y + a2 + b2 
				/ ( y + a3 ))))
	else:
		value = r * math.exp(-y) / ( z + c1 + d1 / ( z + c2 + d2 / ( z + c3 + d3 / ( z + c4 + d4 / ( z + c5 + d5 / ( z + c6 ))))))

	if (not up):
		value = 1.0 - value

	return value

'''
  Purpose:

	PPND produces the normal deviate value corresponding to lower tail area = P.

  Licensing:

	This code is distributed under the GNU LGPL license. 

  Modified:

	03 November 2010
	04 April 2017

  Author:

	Original FORTRAN77 version by J Beasley, S Springer.
	C version by John Burkardt.
	Python version by JeeHang Lee (jeehanglee@gmail.com)

  Reference:

	J Beasley, S Springer,
	Algorithm AS 111:
	The Percentage Points of the Normal Distribution,
	Applied Statistics,
	Volume 26, Number 1, 1977, pages 118-121.

  Parameters:

	Input, P, the value of the cumulative probability
	densitity function.  0 < P < 1.

	Output, don't use in the Python version, integer *IFAULT, error flag.
	0, no error.
	1, P <= 0 or P >= 1.  PPND is returned as 0.

	Output, PPND, the normal deviate value with the property that
	the probability of a standard normal deviate being less than or
	equal to PPND is P.
'''
def ppnd(p):

	# print '__ppnd__'
	a0 = 2.50662823884
	a1 = -18.61500062529
	a2 = 41.39119773534
	a3 = -25.44106049637
	b1 = -8.47351093090
	b2 = 23.08336743743
	b3 = -21.06224101826
	b4 = 3.13082909833
	c0 = -2.78718931138
	c1 = -2.29796479134
	c2 = 4.85014127135
	c3 = 2.32121276858
	d1 = 3.54388924762
	d2 = 1.63706781897
	r = 0.0
	split = 0.42
	value = 0.0

	# 0.08 < P < 0.92
	if math.fabs(p - 0.5) <= split:
		r = ( p - 0.5 ) * ( p - 0.5 )

		value = ( p - 0.5 ) * ( ( ( 
			a3   * r 
		  + a2 ) * r 
		  + a1 ) * r 
		  + a0 ) / ( ( ( ( 
			b4   * r 
		  + b3 ) * r 
		  + b2 ) * r 
		  + b1 ) * r 
		  + 1.0 )
	# P < 0.08 or P > 0.92
	# R = min (P, i - P)
	elif 0.0 < p and p < 1.0:
		if 0.5 < p:
			r = math.sqrt(-math.log(1.0 - p))
		else:
			r = math.sqrt(-math.log(p))

		value = ( ( ( 
				c3   * r 
			  + c2 ) * r 
			  + c1 ) * r 
			  + c0 ) / ( ( 
				d2   * r 
			  + d1 ) * r 
			  + 1.0 )
		
		if p < 0.5:
			value = -value
	# P <= 0.0 or 1.0 <= P
	else:
		value = 0.0

	return value

'''
  Purpose:

	GAMMAD computes the Incomplete Gamma Integral

  Licensing:

	This code is distributed under the GNU LGPL license. 

  Modified:

	13 November 2010
	04 April 2017

  Author:

	Original FORTRAN77 version by B Shea.
	C version by John Burkardt.
	Python version by JeeHang Lee (jeehanglee@gmail.com)

  Reference:

	B Shea,
	Algorithm AS 239:
	Chi-squared and Incomplete Gamma Integral,
	Applied Statistics,
	Volume 37, Number 3, 1988, pages 466-473.

  Parameters:

	Input, X, P, the parameters of the incomplete
	gamma ratio.  0 <= X, and 0 < P.

	Output, not used in the Python version, int IFAULT, error flag.
	0, no error.
	1, X < 0 or P <= 0.

	Output, GAMMAD, the value of the incomplete
	Gamma integral.
'''
def gammad(x, p):
	#print '__gammad__'
	a = 0.0
	an = 0.0
	arg = 0.0
	b = 0.0
	c = 0.0
	elimit = - 88.0
	oflo = 1.0E+37
	plimit = 1000.0
	pn1 = 0.0
	pn2 = 0.0
	pn3 = 0.0
	pn4 = 0.0
	pn5 = 0.0
	pn6 = 0.0
	rn = 0.0
	tol = 1.0E-14
	upper = 0
	value = 0.0
	xbig = 1.0E+08

	# verify the input
	if x < 0.0:
		return value

	if p <= 0.0:
		return value

	if x == 0.0:
		return value

	# if P is large, use a normal approximation
	if plimit < p:
		pn1 = 3.0 * math.sqrt ( p ) * ( math.pow ( x / p, 1.0 / 3.0 ) + 1.0 / ( 9.0 * p ) - 1.0 )
		upper = 0
		value = alnorm ( pn1, upper )
		return value

	# if X is large, set value = 1
	if xbig < x:
		value = 1.0
		return value

	# Use Pearson's series expansion
	if x <= 1.0 or x < p:
		arg = p * math.log ( x ) - x - math.lgamma ( p + 1.0 )
		c = 1.0
		value = 1.0
		a = p

		while True:
			a = a + 1.0
			c = c * x / a
			value = value + c

			if c <= tol:
				break

		arg = arg + math.log(value)

		if elimit <= arg:
			value = math.exp( arg )
		else:
			value = 0.0
	# Use a continued fraction expansion
	else:
		arg = p * math.log ( x ) - x - math.lgamma ( p )
		a = 1.0 - p
		b = a + x + 1.0
		c = 0.0
		pn1 = 1.0
		pn2 = x
		pn3 = x + 1.0
		pn4 = x * b
		value = pn3 / pn4

		while True:
			a = a + 1.0
		  	b = b + 2.0
		  	c = c + 1.0
		  	an = a * c
		  	pn5 = b * pn3 - an * pn1
		  	pn6 = b * pn4 - an * pn2

			if pn6 != 0.0:
				rn = pn5 / pn6

				if math.fabs(value - rn) <= min(tol, tol * rn): #r8_min(tol, tol * rn):
					break

				value = rn

			pn1 = pn3
			pn2 = pn4
			pn3 = pn5
			pn4 = pn6

			# Re-scale terms in continued fraction if terms are large.
			if oflo <= math.fabs(pn5):
				pn1 = pn1 / oflo
				pn2 = pn2 / oflo
				pn3 = pn3 / oflo
				pn4 = pn4 / oflo

		arg = arg + math.log(value)

		if elimit <= arg:
			value = 1.0 - math.exp(arg)
		else:
			value = 1.0

	return value

'''
  AS R85 PPCHI2

  Purpose:

	PPCHI2 evaluates the percentage points of the Chi-squared PDF

  Discussion

	Incorporates the suggested changes in AS R85 (vol.40(1),
	pages 233-5, 1991) which should eliminate the need for the limited
	range for P, though these limits have not been removed
	from the routine.

  Licensing:

	This code is distributed under the GNU LGPL license. 

  Modified:

	03 November 2010

  Author:

	Original FORTRAN77 version by Donald Best, DE Roberts.
	C version by John Burkardt.
	Python version by JeeHang Lee (jeehanglee@gmail.com)

  Reference:

	Donald Best, DE Roberts,
	Algorithm AS 91:
	The Percentage Points of the Chi-Squared Distribution,
	Applied Statistics,
	Volume 24, Number 3, 1975, pages 385-390.

  Parameters:

	Input, P,  value of the chi-squared cumulative
	probability density function.
	0.000002 <= P <= 0.999998.

	Input, V, the parameter of the chi-squared probability
	density function.
	0 < V.

	Input, G, the value of log ( Gamma ( V / 2 ) ).

	Output, int *IFAULT, is nonzero if an error occurred.
	0, no error.
	1, P is outside the legal range.
	2, V is not positive.
	3, an error occurred in GAMMAD.
	4, the result is probably as accurate as the machine will allow.

	Output, PPCHI2, the value of the chi-squared random
	deviate with the property that the probability that a chi-squared random
	deviate with parameter V is less than or equal to PPCHI2 is P.
'''
def ppchi2(p, v, g):
	res = 0.0
	a = 0.0
	b = 0.0
	c = 0.0
	aa = 0.6931471806

	c1 = 0.01
	c2 = 0.222222
	c3 = 0.32
	c4 = 0.4
	c5 = 1.24
	c6 = 2.2
	c7 = 4.67
	c8 = 6.66
	c9 = 6.73
	c10 = 13.32
	c11 = 60.0
	c12 = 70.0
	c13 = 84.0
	c14 = 105.0
	c15 = 120.0
	c16 = 127.0
	c17 = 140.0
	c18 = 175.0
	c19 = 210.0
	c20 = 252.0
	c21 = 264.0
	c22 = 294.0
	c23 = 346.0
	c24 = 420.0
	c25 = 462.0
	c26 = 606.0
	c27 = 672.0
	c28 = 707.0
	c29 = 735.0
	c30 = 889.0
	c31 = 932.0
	c32 = 966.0
	c33 = 1141.0
	c34 = 1182.0
	c35 = 1278.0
	c36 = 1740.0
	c37 = 2520.0
	c38 = 5040.0
	e = 0.5E-06

	p1= 0.0
	p2= 0.0
	q= 0.0
	s1= 0.0
	s2= 0.0
	s3= 0.0
	s4= 0.0
	s5= 0.0
	s6= 0.0
	t= 0.0

	if1_list = []

	pmax = 0.999998
	pmin = 0.000002
	maxit = 20

	value = -1.0
	ch = 0.0
	x = 0.0
	xx = 0.0

	# verify invalid arguments
	if p < pmin or p > pmax:
		return value
	
	if v < 0.0:
		return value

	xx = 0.5 * v
	c = xx - 1.0

	# Starting approximation for small chi-squared
	if v < -c5 * math.log(p):
		ch = math.pow(p * xx * math.exp(g + xx * aa), 1.0 / xx)
		if ch < e:
			value = ch
			return value
	# Starting approximation for V less than or equal to 0.32
	elif v <= c3:
		ch = c4
		a = math.log(1.0 - p)

		while True:
			q = ch
			p1 = 1.0 + ch * (c7 + ch)
			p2 = ch * (c9 + ch * (c8 + ch))
			
			t = - 0.5 + ( c7 + 2.0 * ch ) / p1 - ( c9 + ch * ( c10 + 3.0 * ch ) ) / p2

			ch = ch - (1.0 - math.exp(a + g + 0.5 * ch + c * aa) * p2 / p1) / t

			if math.fabs(q / ch - 1.0) <= c1:
				break
	else:
		# Call to algorithm AS111 - note that P has been tested above.
		# AS 241 could be used as an alternative
		x = ppnd(p)
		
		# starting approximation using Wilson and Hilferty estimate
		p1 = c2 / v
		ch = v * math.pow(x * math.sqrt(p1) + 1.0 - p1, 3)

		# starting approximation for P tending to 1
		if c6 * v + 6.0 < ch:
			ch = -2.0 * (math.log(1.0 - p) - c * math.log(0.5 * ch) + g)

	# Call to algorithm AS 239 and calculation of seven term Taylor series
	for i in range(1, maxit + 1):
		q = ch
		p1 = 0.5 * ch
		p2 = p - gammad( p1, xx)

		'''if (len(if1_list) > 0) and (if1_list[0] != 0):
			return value'''
		
		t = p2 * math.exp(xx * aa + g + p1 - c * math.log(ch))
		b = t / ch
		a = 0.5 * t - b * c
		s1 = ( c19 + a * ( c17 + a * ( c14 + a * ( c13 + a * ( c12 + c11 * a ))))) / c24
		s2 = ( c24 + a * ( c29 + a * ( c32 + a * ( c33 + c35 * a )))) / c37
		s3 = ( c19 + a * ( c25 + a * ( c28 + c31 * a ))) / c37
		s4 = ( c20 + a * ( c27 + c34 * a) + c * ( c22 + a * ( c30 + c36 * a ))) / c38
		s5 = ( c13 + c21 * a + c * ( c18 + c26 * a )) / c37
		s6 = ( c15 + c * ( c23 + c16 * c )) / c38
		ch = ch + t * (1.0 + 0.5 * t * s1 - b * c * (s1 - b * ( s2 - b * ( s3 - b * ( s4 - b * ( s5 - b * s6 ))))))

		if e < math.fabs(q / ch - 1.0):
			value = ch
			return value
	
	value = ch
	return value

'''
  Purpose:

    TRIGAMMA calculates trigamma(x) = d^2 log(gamma(x)) / dx^2

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    08 November 2010
	05 April 2016

  Author:

    Original FORTRAN77 version by BE Schneider.
    C version by John Burkardt.
	Python version by JeeHang Lee (jeehanglee@gmail.com)

  Reference:

    BE Schneider,
    Algorithm AS 121:
    Trigamma Function,
    Applied Statistics, 
    Volume 27, Number 1, pages 97-99, 1978.

  Parameters:

    Input, X, the argument of the trigamma function.
    0 < X.

    Output, int *IFAULT, error flag.
    0, no error.
    1, X <= 0.

    Output, TRIGAMMA, the value of the trigamma function at X.
'''
def trigamma(x):
	a = 0.0001
	b = 5.0
	b2 =  0.1666666667
	b4 = -0.03333333333
	b6 =  0.02380952381
	b8 = -0.03333333333
	value = 0.0
	y = 0.0
	z = 0.0

	# verify the input
	if x <= 0.0:
		# raise the exception
		return value

	z = x

	# use small value approximation if X <= A
	if x <= a:
		value = 1.0 / x / x
		return value

	# increase argument to (X + I)>= B
	while z < b:
		value = value + 1.0 / z / z
		z = z + 1.0

	# apply asymptotic formula if the argument is B or greater
	y = 1.0 / z / z
	value = value + 0.5 * y + ( 1.0 
			+ y * ( b2  
			+ y * ( b4  
			+ y * ( b6  
			+ y *   b8 )))) / z

	return value

'''
## DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    20 March 2016
#	 05 April 2017
#
#  Author:
#
#    Original FORTRAN77 version by Jose Bernardo.
#    Python version by John Burkardt.
#	 Python version updated by JeeHang Lee (jeehanglee@gmail.com)
#
#  Reference:
#
#    Jose Bernardo,
#    Algorithm AS 103:
#    Psi ( Digamma ) Function,
#    Applied Statistics,
#    Volume 25, Number 3, 1976, pages 315-317.
#
#  Parameters:
#
#    Input, real X, the argument of the digamma function.
#    0 < X.
#
#    Output, real DIGAMMA, the value of the digamma function at X.
#
#    Output, integer IFAULT, error flag.
#    0, no error.
#    1, X <= 0.
'''
def digamma ( x ):

	import numpy as np
	nc = 100000000.0

	#
	#  Check the input.
	#
	if ( x <= 0.0 ):
		value = 0.0
		return value
	#
	#  Initialize.
	#
	ifault = 0
	value = 0.0
	#
	#  Use approximation for small argument.
	#
	if ( x <= 0.000001 ):
		euler_mascheroni = 0.57721566490153286060
		value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
		return value, ifault
	#
	#  Reduce to DIGAMA(X + N).
	#
	while ( x < 8.5 ):
		value = value - 1.0 / x
		x = x + 1.0
	#
	#  Use Stirling's (actually de Moivre's) expansion.
	#
	r = 1.0 / x
	value = value + np.log ( x ) - 0.5 * r
	r = r * r
	value = value \
		- r * ( 1.0 / 12.0 \
		- r * ( 1.0 / 120.0 \
		- r * ( 1.0 / 252.0 \
		- r * ( 1.0 / 240.0 \
		- r * ( 1.0 / 132.0 ) ) ) ) )

	# JeeHang's adjustment - coping with discrepancy 
	value = value * nc
	value = round(value)
	value = value / nc

	return value


'''
  Purpose:

    R8MAT_MV_NEW multiplies a matrix times a vector.

  Discussion:

    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
    in column-major order.

    For this routine, the result is returned as the function value.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    11 April 2007

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns of the matrix.

    Input, double A[M,N], the M by N matrix.

    Input, double X[N], the vector to be multiplied by A.

    Output, double R8MAT_MV[M], the product A*X.
'''
def r8mat_mv_new(m, n, a, x):
	i = 0
	j = 0
	y = []

	for i in range(m):
		y.insert(i, 0.0)
		for j in range(n):
			temp = y[i]
			y.insert(i, temp + a[i + j * m] * x[j])

	return y

'''
  Purpose:

    R8VEC_DOT_PRODUCT computes the dot product of a pair of R8VEC's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    26 July 2007

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the vectors.

    Input, double A1[N], A2[N], the two vectors to be considered.

    Output, double R8VEC_DOT_PRODUCT, the dot product of the vectors.
'''
def r8vec_dot_product(n, a1, a2):
	i = 0
	value = 0.0
	
	for i in range(n):
		value = value + a1[i] * a2[i]

	return value
