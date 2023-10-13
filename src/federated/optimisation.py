import numpy as np 
import tensorflow as tf 


def gamma_gradients_step(gamma, Z, knots_y, learning_rate, n_epochs):

	def _loss_gamma():
		return tf.reduce_sum(tf.square(knots_y - Z @ gamma), axis=0)

	gamma = tf.Variable(gamma, dtype=tf.float32) 

	Z = tf.cast(Z, dtype=tf.float32)
	knots_y = tf.cast(knots_y, dtype=tf.float32)

	#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	loss_gamma = []
	for _ in range(n_epochs):

		with tf.GradientTape() as tape:
			mse = _loss_gamma()

		return tape.gradient(mse, gamma), np.mean(mse.numpy())

		#optimizer.minimize(_loss_gamma, [gamma])
		#loss_gamma.append(_loss_gamma().numpy())

	#return gamma.numpy(), loss_gamma


def beta_gradients_step(beta_init, X, S, dS, delta, learning_rate, n_epochs):

	def _loss_beta():
		
	    nu = S + tf.matmul(X, beta)
	    log_likelihood = delta * (logdS + nu) - tf.exp(nu)
	    return -1.0 * tf.reduce_sum(log_likelihood, axis=0)

	beta = tf.Variable(beta_init.copy(), dtype=tf.float32)

	X = tf.cast(X, dtype=tf.float32)
	S = tf.cast(S, dtype=tf.float32)
	logdS = tf.cast(tf.math.log(dS), dtype=tf.float32)
	delta = tf.cast(delta, dtype=tf.float32)

	#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	loss_beta = []
	for _ in range(n_epochs):

		with tf.GradientTape() as tape:
			nll = _loss_beta()

		return tape.gradient(nll, beta), nll.numpy()

		#optimizer.minimize(_loss_beta, [beta])
		#loss_beta.append(np.mean(_loss_beta().numpy()))

	#return optimizer.compute_gradients(_loss_beta, [beta_init])[0][1]


def beta_hessian(beta, X, S, dS, delta):
	# stackoverflow: hessian matrix of a keras model with tf.hessians

	def _loss_beta():
		
	    nu = S + tf.matmul(X, beta)
	    log_likelihood = delta * (tf.math.log(dS) + nu) - tf.exp(nu)
	    return -1.0 * tf.reduce_sum(log_likelihood, axis=0)
	
	beta = tf.Variable(beta, dtype=tf.float32)
	
	X = tf.cast(X, dtype=tf.float32)
	S = tf.cast(S, dtype=tf.float32)
	dS = tf.cast(dS, dtype=tf.float32)
	delta = tf.cast(delta, dtype=tf.float32)

	with tf.GradientTape(persistent=True) as tape_outer:
		tape_outer.watch(beta)

		with tf.GradientTape() as tape_inner:
			tape_inner.watch(beta)

			loss = _loss_beta()

		gradient = tape_inner.gradient(loss, beta)
	return tape_outer.jacobian(gradient, beta)



def gamma_gradients_steps(gamma, Z, knots_y, learning_rate, n_epochs):

	for _ in range(n_epochs-1):
		pass

	return 
	

def beta_gradients_steps(beta_init, X, S, dS, delta, learning_rate, n_epochs):

	for _ in range(n_epochs-1):
		pass

	return 