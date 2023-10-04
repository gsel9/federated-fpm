import scipy as sp 
import numpy as np 
import tensorflow as tf 

from sklearn.preprocessing import StandardScaler

from splines import NaturalCubicSpline, knots


def centralised_benchmark(X, delta, logtime, n_knots, gamma_init, beta_init, knots_x, knots_y,
						  learning_rate, global_epochs, order, intercept):
	
	# NOTE: assume local and global standardisation is irrelevant 
	#scaler = StandardScaler()
	#X = scaler.fit_transform(X)

	#knots_x, knots_y = knots(logtime, delta, n_knots)

	ncs = NaturalCubicSpline(knots=knots_x, order=order, intercept=intercept)
	Z = ncs.transform(knots_y, derivative=False)
	dZ = ncs.transform(logtime, derivative=True)
	Z_long = ncs.transform(logtime, derivative=False)

	gamma, loss_gamma = fit_gamma(gamma_init, Z, knots_y, global_epochs, learning_rate)

	# spline matrices 
	S = Z_long @ gamma
	dS = dZ @ gamma[1:]

	beta, loss_beta = fit_beta(beta_init, X, S, dS, delta, global_epochs, learning_rate)

	return gamma, beta, loss_gamma, loss_beta


def fit_gamma(gamma, Z, knots_y, epochs, learning_rate, order=1, intercept=True):

	def _loss_gamma():
		return tf.reduce_sum(tf.square(knots_y - Z @ gamma), axis=0)

	Z = tf.cast(Z, dtype=tf.float32)
	y_true = tf.cast(knots_y, dtype=tf.float32)
	learning_rate = tf.cast(learning_rate, dtype=tf.float32)

	gamma = tf.Variable(gamma, dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	losses = []
	for epoch in range(epochs):

		optimizer.minimize(_loss_gamma, [gamma])
		losses.append(np.mean(_loss_gamma()))

	return gamma.numpy(), losses


def fit_beta(beta, X, S, dS, delta, epochs, learning_rate):

	def _loss_beta():
		
	    nu = S + tf.matmul(X, beta)
	    log_likelihood = delta * (tf.math.log(dS) + nu) - tf.exp(nu)
	    return -1.0 * tf.reduce_sum(log_likelihood, axis=0)

	beta = tf.Variable(beta, dtype=tf.float32)

	X = tf.cast(X, dtype=tf.float32)
	S = tf.cast(S, dtype=tf.float32)
	dS = tf.cast(dS, dtype=tf.float32)
	delta = tf.cast(delta, dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	losses = []
	for epoch in range(epochs):

		optimizer.minimize(_loss_beta, [beta])
		losses.append(np.mean(_loss_beta()))

	return beta.numpy(), losses
