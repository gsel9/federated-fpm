import scipy as sp 
import numpy as np 
import tensorflow as tf 

#from sklearn.preprocessing import StandardScaler

from splines import NaturalCubicSpline, knots
from optimisation import gamma_gradients_step, beta_gradients_step


def create_client(X, delta, logtime, n_epochs, learning_rate, knots_x, knots_y, n_knots=6, seed=42):

	print("Frac censored:", sum(delta) / delta.size)

	# NOTE: assume no need for local standardisation 
	#scaler = StandardScaler()
	#X = scaler.fit_transform(X)

	#knots_x, knots_y = knots(logtime, delta, n_knots)

	ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)

	Z = ncs.transform(knots_y, derivative=False)
	Z_long = ncs.transform(logtime, derivative=False)

	dZ = ncs.transform(logtime, derivative=True)

	return Client(X, Z, dZ, Z_long, delta, logtime, knots_y,
				  n_epochs, learning_rate)


class Client:

	def __init__(self, X, Z, dZ, Z_long, delta, logtime, knots_y, 
				 n_epochs, learning_rate):

		self.X = X 
		self.Z = Z 
		self.dZ = dZ 
		self.Z_long = Z_long
		self.delta = delta
		self.logtime = logtime
		self.knots_y = knots_y

		self.n_epochs = n_epochs
		self.learning_rate = learning_rate

		self.S, self.dS = None, None 
		self.beta, self.gamma = None, None 
		self.loss_beta, self.loss_gamma = [], []

	@property
	def n_samples(self):
		return self.X.shape[0 ]

	@property 
	def z_statistic(self):
		# return each beta coefficient divided by its standard error 
		return self.beta / self.beta_se

	def gamma_step(self):

		gradients, loss_gamma = gamma_gradients_step(self.gamma, self.Z, self.knots_y, 
		  		 									     self.learning_rate, self.n_epochs)
		return gradients, loss_gamma

	def beta_step(self):

		gradients, loss_beta = beta_gradients_step(self.beta, self.X, self.S, self.dS, self.delta,
										  			   self.learning_rate, self.n_epochs)
		return gradients, loss_beta
		
	def update_weights(self, gamma=None, beta=None):

		if gamma is not None:
			self.gamma = gamma 
	
		if beta is not None:
			self.beta = beta 

	def update_splines(self):

		# update spline matrices 
		self.S = self.Z_long @ self.gamma
		self.dS = self.dZ @ self.gamma[1:]
		
		if np.min(self.dS) < 0:
			raise ValueError("Negative values in dS")
