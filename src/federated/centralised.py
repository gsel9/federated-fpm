import numpy as np 
import tensorflow as tf 

from splines import NaturalCubicSpline, knots


# TODO: Compare beta and spline coefficients from centralised and FL models
# also, loss functions (and cmats?)
def centralised_benchmark(X, delta, logtime, order=1, intercept=True, seed=42):

	knots_x, knots_y = knots(logtime, delta)

	ncs = NaturalCubicSpline(knots=knots_x, order=order, intercept=intercept)
	Z = ncs.transform(knots_y, derivative=False)
	dZ = ncs.transform(logtime, derivative=True)
	Z_long = ncs.transform(logtime, derivative=False)

	initializer = tf.keras.initializers.GlorotNormal(seed=seed)
	beta = initializer(shape=(X.shape[1], 1))
	gamma = initializer(shape=(Z.shape[1], 1))

	gamma, loss_gamma = fit_gamma(gamma, Z, knots_y)

	# spline matrices 
	S = Z_long @ gamma
	dS = dZ @ gamma[1:]

	beta, loss_beta = fit_beta(beta, X, S, dS, delta)

	return gamma, beta, loss_gamma, loss_beta


def fit_gamma(gamma, Z, knots_y, epochs=200, order=1, intercept=True):

	def mse_loss():

	    y_pred = tf.matmul(Z, gamma)
	    return loss_object(y_true=y_true, y_pred=y_pred)

	Z = tf.cast(Z, dtype=tf.float32)
	y_true = tf.cast(knots_y, dtype=tf.float32)

	gamma = tf.Variable(gamma, dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=1)
	loss_object = tf.keras.losses.MeanSquaredError()

	losses = []
	for epoch in range(epochs):
	    
	    optimizer.minimize(mse_loss, [gamma])
	    losses.append(mse_loss())

	return gamma.numpy(), losses


def fit_beta(beta, X, S, dS, delta, epochs=200):
	
	def neg_log_likelihood():
	    
	    nu = S + tf.matmul(X, beta)
	    log_likelihood = delta * (tf.math.log(dS) + nu) - tf.exp(nu)

	    return -1.0 * log_likelihood

	beta = tf.Variable(beta, dtype=tf.float32)

	X = tf.cast(X, dtype=tf.float32)
	S = tf.cast(S, dtype=tf.float32)
	dS = tf.cast(dS, dtype=tf.float32)
	delta = tf.cast(delta, dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=1)

	losses = []
	for epoch in range(epochs):
	    
	    optimizer.minimize(neg_log_likelihood, [beta])
	    losses.append(np.mean(neg_log_likelihood().numpy()))

	return beta.numpy(), losses
