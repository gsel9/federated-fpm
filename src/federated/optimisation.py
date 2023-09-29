import numpy as np 
import tensorflow as tf 


def estimate_gamma_gradients(gamma, Z, knots_y, learning_rate, n_epochs):

	def _loss_gamma():
		return tf.reduce_mean(tf.square(knots_y - Z @ gamma))

	gamma = tf.Variable(gamma, dtype=tf.float32) 

	Z = tf.cast(Z, dtype=tf.float32)
	knots_y = tf.cast(knots_y, dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	loss_gamma = []
	for _ in range(n_epochs):

		with tf.GradientTape() as tape:
			mse = _loss_gamma()

		return tape.gradient(mse, gamma)

		optimizer.minimize(_loss_gamma, [gamma])
		loss_gamma.append(_loss_gamma().numpy())

	return gamma.numpy(), loss_gamma


def estimate_beta_gradients(beta, X, S, dS, delta, learning_rate, n_epochs):

	def _loss_beta():
		    
	    nu = S + tf.matmul(X, beta)
	    log_likelihood = delta * (tf.math.log(dS) + nu) - tf.exp(nu)
	    return -1.0 * log_likelihood

	beta = tf.Variable(beta, dtype=tf.float32)

	X = tf.cast(X, dtype=tf.float32)
	S = tf.cast(S, dtype=tf.float32)
	dS = tf.cast(dS, dtype=tf.float32)
	delta = tf.cast(delta, dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	loss_beta = []
	for _ in range(n_epochs):

		with tf.GradientTape() as tape:
			nll = _loss_beta()

		return tape.gradient(nll, beta)

		#optimizer.minimize(_loss_beta, [beta])
		#loss_beta.append(np.mean(_loss_beta().numpy()))

	return beta.numpy(), beta


def gradient_descent_gamma(gamma, Z, knots_y, learning_rate, n_epochs):

	def _loss_gamma():
		return loss_object(y_true=knots_y, y_pred=Z @ gamma)

	Z = tf.cast(Z, dtype=tf.float32)
	knots_y = tf.cast(knots_y, dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	loss_object = tf.keras.losses.MeanSquaredError()

	loss_gamma = []
	for _ in range(n_epochs):

	    optimizer.minimize(_loss_gamma, [gamma])
	    loss_gamma.append(_loss_gamma().numpy())

	return gamma.numpy(), loss_gamma


def gradient_descent_beta(beta, X, S, dS, delta, learning_rate, n_epochs):

	def _loss_beta():
		    
	    nu = S + tf.matmul(X, beta)
	    log_likelihood = delta * (tf.math.log(dS) + nu) - tf.exp(nu)
	    return -1.0 * log_likelihood

	beta = tf.Variable(beta, dtype=tf.float32)

	X = tf.cast(X, dtype=tf.float32)
	S = tf.cast(S, dtype=tf.float32)
	dS = tf.cast(dS, dtype=tf.float32)
	delta = tf.cast(delta[:, None], dtype=tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	loss_beta = []
	for _ in range(n_epochs):
	
	    optimizer.minimize(_loss_beta, [beta])
	    loss_beta.append(np.mean(_loss_beta().numpy()))

	return beta.numpy(), loss_beta