import scipy as sp 
import numpy as np 
import tensorflow as tf 

from sklearn.preprocessing import StandardScaler

from splines import NaturalCubicSpline, knots
from optimisation import (solve_gamma, 
						  solve_beta,
						  compute_gamma_gradients,
						  compute_beta_gradients)


def create_client(X, delta, logtime, n_epochs, learning_rate, seed=42):

	print("Frac censored:", sum(delta) / delta.size)

	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	knots_x, knots_y = knots(logtime, delta)

	ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)

	Z = ncs.transform(knots_y, derivative=False)
	Z_long = ncs.transform(logtime, derivative=False)

	dZ = ncs.transform(logtime, derivative=True)

	initializer = tf.keras.initializers.GlorotNormal(seed=seed)
	beta = initializer(shape=(X.shape[1], 1))
	gamma = initializer(shape=(Z.shape[1], 1))

	return Client(X, Z, dZ, Z_long, delta, logtime, knots_y,
				  beta, gamma, n_epochs, learning_rate)


class Client:

	def __init__(self, X, Z, dZ, Z_long, delta, logtime, knots_y, 
				 beta, gamma, n_epochs, learning_rate=1):

		self.X = X 
		self.Z = Z 
		self.dZ = dZ 
		self.Z_long = Z_long
		self.delta = delta
		self.logtime = logtime
		self.knots_y = knots_y

		self.beta = beta 
		self.gamma = gamma 
		self.n_epochs = n_epochs
		self.learning_rate = learning_rate

		self.S, self.dS = None, None 
		self.loss_beta, self.loss_gamma = [], []

	def fit_gamma(self):

		self.gamma, loss_gamma = solve_gamma(self.gamma, self.Z, self.knots_y, 
											 self.learning_rate, self.n_epochs)
		self.loss_gamma.extend(loss_gamma)
		
	def fit_beta(self):
		
		self.beta, loss_beta = solve_beta(self.beta, self.X, self.S, self.dS, self.delta,
										  self.learning_rate, self.n_epochs)
		self.loss_beta.extend(loss_beta)

	def gamma_gradients(self, gamma_variable):

		gradients = compute_gamma_gradients(gamma_variable, self.Z, self.knots_y)
		return gradients 

	def beta_gradients(self, beta_variable):

		dl_db, d2l_db2 = compute_beta_gradients(beta_variable, self.S, self.X, self.delta, self.dS)
		return dl_db, d2l_db2
		#gradients = compute_beta_gradients(beta_variable, self.S, self.X, self.delta, self.dS)
		#return gradients 

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
