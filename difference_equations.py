#!/usr/bin/python2.7
'''
This file contains tools to simulate range expansions using coupled difference equations. 
Author:
Ching-Hao Wang and Kirill Korolev
'''

from __future__ import division
import numpy as np
from scipy import stats
from scipy.optimize import fminbound
import datetime # for timestamps
import os as os
import cPickle as pickle
import sympy

#*****************
# simulation class
#*****************
class population:
	'''
	This class runs range expansion simulations and performs basic analysis including velocity computation and front shape visualization.
	'''
	def __init__(self, growth_type='linear_constant', migration = 0.5, parameters={'low_growth_rate_r':1.5, 'carrying_capacity':1, 'threshold_cs':0.5}, time=0, L=100, centering=2.0/5.0, equilibrate=True, EPSILON=10**-3):
		self.L = L
		self.centering = centering
		self.time = time
		self.m = migration
		self.EPSILON = EPSILON
		if growth_type == 'linear_constant':
			self.grow = self.gf_linear_constant # growth function
			self.r = parameters['low_growth_rate_r']
			self.K = parameters['carrying_capacity']
			self.cs = parameters['threshold_cs']			
		elif growth_type == 'logistic':
			self.grow = self.gf_logistic # growth function
			self.r = parameters['low_growth_rate_r']
			self.logistic_K = parameters['K']
			self.K = self.logistic_K * (1-1/self.r) # stable fixed point; if stable
		elif growth_type == 'cubic':
			self.grow = self.gf_cubic # growth function
			self.r = parameters['low_growth_rate_r']
			self.b = parameters['b']
			self.d = parameters['d']
			self.K = (self.b + np.sqrt(self.b*self.b + 4*self.d*(self.r-1)))/(2*self.d)
		elif growth_type == 'hill':
			self.grow = self.gf_hill
			self.A = parameters['A']
			self.B = parameters['B']
			self.n = parameters['n']
			self.K = 1 + self.A + self.B
			for i in xrange(1000):
				self.K = self.grow(self.K)
		elif growth_type == 'beverton_holt': # from incorporating Allee effects in fish stock-recruitment models and applications for determining reference points by chen irvine and cass.
			self.grow = self.gf_beverton_holt
			self.A = parameters['A']
			self.B = parameters['B']
			self.offset = parameters['offset']
			self.K = 1 + self.A + self.B + self.offset
			for i in xrange(1000):
				self.K = self.grow(self.K)
		else:
			assert(growth_type == 'linear_constant')

		self.pop = self.K*np.ones(self.L)
		self.pop[np.round(self.centering*self.L):] = 0 # set up population
		self.shift = 0 # no shifting yet
		self.target_N_tot = np.sum(self.pop) # target population size
		
		if equilibrate:
			#self.simulate(T=int(np.round(2*self.L*self.L/(self.m+self.EPSILON))))
			self.simulate(T=10**4) # simple and fast
			self.measure_velocity()
			#self.measure_front_shape()
		self.equilibrate = equilibrate

	def gf_beverton_holt(self, c=None):
		'''
		Michaelis-Menten with offset.
		'''
		if c is None:
			self.pop = (self.pop - self.offset) * (self.pop > self.offset)
			self.pop = (self.pop > 0) * self.A * self.pop / (self.B + self.pop)
		else:
			cmo = (c - self.offset) * (c > self.offset)
			return self.A * cmo / (self.B + cmo)

	def gf_hill(self, c=None):
		'''
		Discrete map based on a Hill function.

		'''
		if c is None:
			self.pop = self.A*self.pop**self.n/(self.B+self.pop**self.n)
		else:
			return self.A*c**self.n/(self.B+c**self.n)

	def gf_logistic(self, c=None):
		'''
		Discrete logistic map.
		'''
		if c is None:
			self.pop = self.r*self.pop*(1-self.pop/self.logistic_K)
		else:
			return self.r*self.pop*(1-self.pop/self.logistic_K)

	def gf_cubic(self, c=None):
		'''
		Discrete cubic map c at t+1 is rc+bc**2-dc**3.
		'''
		if c is None:
			self.pop = self.pop*(self.r + self.b*self.pop - self.d*self.pop*self.pop)
		else:
			return c*(self.r + self.b*c - self.d*c*c)

	def gf_linear_constant(self, c=None):
		'''
		The growth is linear below cs and saturates to the carrying capacity above cs.
		'''
		if c is None:
			self.pop = self.r*self.pop*(self.pop<=self.cs) + self.K*(self.pop>self.cs)
		else:
			return self.r*c*(c<=self.cs) + self.K*(c>self.cs)

	def migrate(self):
		'''
		Performs one migration update.
		'''
		second = self.pop[2]
		one_but_last = self.pop[-2]
		m = self.m
		self.pop[1:-1] = (1-m)*self.pop[1:-1] + m/2*self.pop[:-2] + m/2*self.pop[2:]
		self.pop[0] = (1-m/2)*self.pop[0] + m/2*second
		self.pop[-1] = (1-m/2)*self.pop[-1] + m/2*one_but_last
	
	def perform_shift(self):
		'''
		Performs shift of the simulation box to match the target population size.
		'''
		shift_distance = np.round( (np.sum(self.pop)-self.target_N_tot)/self.K )
		if shift_distance > 0:
			self.pop = np.concatenate( ( self.pop[shift_distance:], np.zeros(shift_distance) ), axis=0)
		if shift_distance < 0:
			self.pop = np.concatenate( ( self.K*np.ones(np.abs(shift_distance)), self.pop[:shift_distance] ), axis=0)
		self.shift += shift_distance

	def simulate(self, T=1):
		'''
		Simulates population updates for T steps.
		'''
		for i in xrange(T):
			self.migrate()
			self.grow()
			if i % np.trunc(self.L/10) == 0: self.perform_shift()
		self.time += T
	
	def measure_velocity(self, N=100, step=None):
		'''
		Estimates expansion velocity from simulations.
		'''
		#if step is None: step = int(100 + np.trunc(self.L/(self.m+self.EPSILON)))
		if step is None: step = 100 # simple and fast
		times = np.zeros(N)
		distances = np.zeros(N)
		times[0] = self.time
		distances[0] = self.shift + np.sum(self.pop)/self.K
		#times = [self.time]
		#distances = [self.shift + np.sum(self.pop)/self.K]
		for i in xrange(1,N):
			self.simulate(step)
			times[i] = self.time
			distances[i] = self.shift + np.sum(self.pop)/self.K
			#times.append(self.time)
			#distances.append(self.shift + np.sum(self.pop)/self.K)
		ones_array = np.ones(times.size)
		A = np.vstack([times, ones_array]).T
		self.velocity, temp = np.linalg.lstsq(A, distances)[0]
		#self.velocity = stats.linregress(times, distances)[0]
		return self.velocity

	def measure_front_shape(self, N=10**3):
		'''
		Returns two arrays with positions and densities.
		'''
		self.c = np.zeros((N, self.L))
		self.x = np.zeros((N, self.L))
		positions = np.arange(self.L)
		for i in xrange(N):
			self.simulate()
			self.c[i,:] = self.pop
			self.x[i,:] = self.shift + positions - self.velocity*i
		
		return (self.x, self.c)

#*****************
# batch tools
#*****************
def create_name(growth_type, r=None, base_part=''):
	'''This function creates a unique name based on the low-density growth rate and growth type. base_part can add an arbitrary name to the beginning.'''
	if r is not None:
		str_r = '_r_' + str(int(np.round(r*100)))
	else:
		str_r = '_time_' + str(datetime.datetime.now())

	return base_part + growth_type + str_r


def migration_sweep(growth_type='cubic', migration_rates=np.linspace(0.01,0.5,100), parameters={'low_growth_rate_r':0.5, 'b':0.35, 'd':0.05}, base_name='', dir_name='./', L=100):
	'''
	Computes the velocities of range expansions over a range of m values and saves the results.
	'''
	v = np.zeros(migration_rates.size)
	for index in xrange(migration_rates.size):
		sim = population(growth_type=growth_type, migration=migration_rates[index], parameters=parameters, L=L)
		v[index] = sim.velocity
	file_name = create_name(growth_type, base_part=base_name) 
	name = os.path.join(dir_name, file_name)
	accuracy = 'simple 10 to the 4 for equilibration and measurement'
	pickle.dump( {'migration_rates':migration_rates, 'velocities':v, 'growth_type':growth_type, 'parameters':parameters, 'L':L, 'accuracy':accuracy}, open(name, 'wb') )
	return (migration_rates, v, file_name)


def find_plateau(migration_rates, velocities, target_v=0.5, tolerance=10**-2):
	'''
	Returns (index_migration_min, index_migration_max) which the smallest range such that the ends do deviate from the target velocity and the plateau is contained within the range. If such elements cannot be found, then three elements that bracket the plateau are returned.
	'''
	assert (velocities[0]<target_v) and (velocities[-1]>target_v) # v_target is in the range

	left = np.searchsorted(velocities, target_v*(1-tolerance), side='left')
	if left>0: left-=1
	right = np.searchsorted(velocities, target_v*(1+tolerance), side='right')
	if right >= migration_rates.size: right = migration_rates.size-1
	
	return (left, right)

def find_plateau_left(migration_rates, velocities, target_v=0.5, tolerance=10**-2):
	'''
	Returns (index_migration_min, index_migration_max) which the smallest range such that the ends do deviate from the target velocity and the plateau is contained within the range. If such elements cannot be found, then three elements that bracket the plateau are returned. Special case when plateau extends to m=0.5
	'''
	assert (velocities[0]<target_v) # v_target is in the range

	left = np.searchsorted(velocities, target_v*(1-tolerance), side='left')
	if left>0: left-=1
	right = migration_rates.size - 1 # the last element
	
	return (left, right)


def refine_plateau(file_name, target_v=0.5, dir_name='./', tolerance=10**-2):
	'''
	In the specified file, finds the values of migration that are consistent with target_v and zooms into that region.
	'''
	data = pickle.load(open(os.path.join(dir_name, file_name), 'rb'))
	(m_left, m_right) = find_plateau(data['migration_rates'], data['velocities'], target_v=target_v, tolerance=tolerance)
	return migration_sweep(growth_type=data['growth_type'], migration_rates=np.linspace(data['migration_rates'][m_left],data['migration_rates'][m_right],100), parameters=data['parameters'], L=data['L'], base_name='refined_'+file_name, dir_name=dir_name)

def refine_plateau_left(file_name, target_v=0.5, dir_name='./', tolerance=10**-2):
	'''
	In the specified file, finds the values of migration that are consistent with target_v and zooms into that region. Special case when plateau extends up to m=0.5
	'''
	data = pickle.load(open(os.path.join(dir_name, file_name), 'rb'))
	(m_left, m_right) = find_plateau_left(data['migration_rates'], data['velocities'], target_v=target_v, tolerance=tolerance)
	return migration_sweep(growth_type=data['growth_type'], migration_rates=np.linspace(data['migration_rates'][m_left],data['migration_rates'][m_right],100), parameters=data['parameters'], L=data['L'], base_name='refined_'+file_name, dir_name=dir_name)

def measure_plateau(growth_type='hill', parameters={'A':7, 'B':1, 'n':2}, target_v=0.5, base_name='', dir_name='./', L=100):
	'''
	Returns the low and upper bounds on the plateau.
	'''
	# find the starting interval for m
	m = 10**-2
	flag = True
	while flag:
		sim = population(growth_type=growth_type, migration=m, parameters=parameters, L=L)
		if sim.velocity < target_v:
			flag = False
		else:
			m /= 10		
	# perform initial sweep
	migration_rates = np.concatenate((np.logspace(np.log(m), -1, 50), np.linspace(0.100001,0.5,50)))
	(m, v, f) = migration_sweep(growth_type=growth_type, migration_rates=migration_rates, parameters=parameters, L=L, dir_name=dir_name)
	# check that the upper bound can be satisfied
	#sim = population(growth_type=growth_type, migration=0.5, parameters=parameters, L=L)
	#assert sim.velocity > target_v # issues will occur when the growth rate is such that the plateau at target_v is the last plateau; although plateau size can in principle be still defined, this function will quit with an error
	v_final = v[-1]
	if (v_final > target_v*(1-10**-4)) and (v_final < target_v*(1+10**-4)): # the plateau extend up to the maximal migration rate
		(m, v, f) = refine_plateau_left(f, target_v=target_v, dir_name=dir_name, tolerance=10**-2)
		(idx_m_min, idx_m_max) = find_plateau_left(m, v, target_v=target_v, tolerance=10**-5)
		if idx_m_max - idx_m_min < 5:
			(m, v, f) = refine_plateau_left(f, target_v=target_v, dir_name=dir_name, tolerance=10**-4)
			(idx_m_min, idx_m_max) = find_plateau_left(m, v, target_v=target_v, tolerance=10**-5)
			if idx_m_max - idx_m_min < 2: return (m[idx_m_min], m[idx_m_max]) # value could be returned here
		(ml, vl, fl) = migration_sweep(growth_type=growth_type, migration_rates=np.linspace(m[idx_m_min],m[idx_m_min+1],10), parameters=parameters, L=L, dir_name=dir_name)
		(idx_m_min, idx_m_max) = find_plateau_left(ml, vl, target_v=target_v, tolerance=10**-5)
		return (np.mean(np.array([ml[idx_m_min],ml[idx_m_min+1]])), 0.5) # value could be returned here
	elif v_final < target_v: return (0, 0) # value could be returned here; plateau is not present
	# plateau is in the interior
	(m, v, f) = refine_plateau(f, target_v=target_v, dir_name=dir_name, tolerance=10**-2)
	(idx_m_min, idx_m_max) = find_plateau(m, v, target_v=target_v, tolerance=10**-5)
	if idx_m_max - idx_m_min < 5:
		(m, v, f) = refine_plateau(f, target_v=target_v, dir_name=dir_name, tolerance=10**-4)
		(idx_m_min, idx_m_max) = find_plateau(m, v, target_v=target_v, tolerance=10**-5)
		if idx_m_max - idx_m_min < 2: return (m[idx_m_min], m[idx_m_max]) # value could be returned here
	(ml, vl, fl) = migration_sweep(growth_type=growth_type, migration_rates=np.linspace(m[idx_m_min],m[idx_m_min+1],10), parameters=parameters, L=L, dir_name=dir_name)
	(mr, vr, fr) = migration_sweep(growth_type=growth_type, migration_rates=np.linspace(m[idx_m_max-1],m[idx_m_max],10), parameters=parameters, L=L, dir_name=dir_name)
	m = np.concatenate((ml,mr))
	v = np.concatenate((vl,vr))
	#print(m)
	#print(v)
	(idx_m_min, idx_m_max) = find_plateau(m, v, target_v=target_v, tolerance=10**-5)
	return (np.mean(np.array([m[idx_m_min],m[idx_m_min+1]])), np.mean(np.array([m[idx_m_max],m[idx_m_max-1]])))






#*****************
# theory tools
#*****************

def linear_velocity(r, m, l=None):
	'''
	Computes the velocity assuming it is a pulled wave. If l is given, uses the relationship between l and v.
	'''
	if r<1: return 0 # linear approximation cannot work

	if l is None:
		def func_to_minimize(l):
			return linear_velocity(r, m, l=l)
		#return func_to_minimize(fminbound(func_to_minimize, 0.01, 100))
		l = fminbound(func_to_minimize, 0.01, 100)
		return (l, func_to_minimize(l))

	else:
		return np.log( r*( 1 - m + m*(np.exp(l)+np.exp(-l))/2 ) ) / l


def plateau_size(r, cs, K=1.0, limit_at_half=True):
	'''
	Returns minimal and maximal migration rates consistent with the plateau at v=1/2. When limit_at_half is set to True, m_max will be limited to 0.5.
	'''
	cs = float(cs)/float(K)
	if r*cs >= K : return (0,0) # no Allee effect, so no locking

	# this determines the position of the separation point between the two roots of lambda(m) 
	if r>1: # if r<=1 there is only one root
		def m_y(y):
			''' The function to be compared to the value of m. If m is greater than the maximum of this function, then there is no solution for lambda at this value of m.'''
			return - (2.0/r) * ( (1+y)**2.0 * (y+1-r) ) / ( (2+y)**2.0 * y**2.0 )
		# condition on existence
		y_max = fminbound(m_y, 10**-3, 1000, xtol=10**-7, maxfun=10**4) # the negative of this has a maximum
		x_max = y_max + 1
	# First we compute the maximum allowed migration, which can come from two conditions: 
	# (i) the density of the second to carrying capacity point takes two generations to reach 
	#     the carrying capacity and 
	# (ii) exponential solution with exp(-lx) at the front exists for v=0.5. 
	# There is also the third requirement that the density never goes below K once it has reached 
	# that value once.

	# condition that the density at "odd points and odd generations" is below the threshold
	m_threshold = 2*cs*(1-cs*r*r)/(1-(r*cs)**2)**2

	# existence condition can only play a role when r>1	
	if r>1:
		m_existence = -m_y(y_max)
		# root checking
		y_root = 1/(cs*r) - 1
		if y_root < y_max:
			# m_threshold is for the wrong root
			m_max = m_existence
		else:
			# m_threshold is for the right root
			m_max = m_threshold
	else:
		m_max = m_threshold

	# condition on nondecreasing density 
	if m_max > 2*(1-cs): m_max = 2*(1-cs)

	# condition on cutoff at m=0.5
	if limit_at_half: 
		m_max = m_max if m_max<0.5 else 0.5

	# Second we compute m_min by requiring that migration is great enough to push the odd point across the threshold

	#We will need to compare the roots x found below to the y_max from the above procedure in order to figure out root switching. 
	# condition b analytically
	#x = sympy.Symbol('x')
	#s_r = sympy.Symbol('r')
	#s_cs = sympy.Symbol('cs')
	#ans_x = sympy.solve((x**3)*(s_cs*s_r/(1-s_cs*s_r)) - (x**2)*(1+s_r/(1-s_cs*s_r)) + x + 1, x)
	#ans_m = [2/(ax-1)*(1-s_cs*s_r)/s_r for ax in ans_x]
	#m_b = [m.subs([(s_r,r), (s_cs,cs)]) for m in ans_m]

	# condition b numerically
	x = np.roots((cs*r/(1-cs*r), -1-r/(1-cs*r), 1, 1))
	x_good = []
	for current_x in x:
		if (abs(np.imag(current_x)) < 10**-5) and (np.real(current_x)>1): x_good.append(np.real(current_x))
	if x_good:
		x_good = max(x_good)
		m_min = 2/(x_good-1)*(1-cs*r)/r
		if r>1: # do root checking
			if x_good < x_max: m_min = m_existence  
	else: # this branch should not occur
		#m_threshold = 1.0
		m_min = m_max

	m_min = m_min if m_min <= 0.5 else 0.5
	
	return (m_min, m_max)
	

def linear_velocity_DSCT(r, m, cs, K, l=None):
	'''
	Computes the velocity assuming it is a pulled wave. If l is given, uses the relationship between l and v.
	'''
	if r<1: return 0 # linear approximation cannot work

	if l is None:
		def func_to_minimize(l):
			return linear_velocity_DSCT(r, m, cs, K, l=l)
		#return func_to_minimize(fminbound(func_to_minimize, 0.01, 100))
		l = fminbound(func_to_minimize, 0.01, 100)
		return (l, func_to_minimize(l))

	else:
		return (r/K*cs + m* (1+ np.cosh(l)))/ l


