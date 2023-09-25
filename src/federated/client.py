import numpy as np 
import tensorflow as tf 

from splines import NaturalCubicSpline, knots


def create_client(X, delta, logtime, n_epochs, learning_rate, seed=42):

	knots_x, knots_y = knots(logtime, delta)

	ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)
	Z = ncs.transform(knots_y, derivative=False)
	dZ = ncs.transform(logtime, derivative=True)
	Z_long = ncs.transform(logtime, derivative=False)

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

		self.loss_beta, self.loss_gamma = [], []

	def record_loss(self):
		
		nu = self.S + tf.matmul(self.X, self.beta)
		loss_value = -1.0 * np.mean(self.delta * (np.log(self.dS) + nu) - tf.exp(nu))

		self.losses.append(loss_value)

	def fit_gamma(self):

		def _loss_gamma():
		    return loss_object(y_true=knots_y, y_pred=Z @ gamma)

		gamma = tf.Variable(self.gamma, dtype=tf.float32)

		Z = tf.cast(self.Z, dtype=tf.float32)
		knots_y = tf.cast(self.knots_y, dtype=tf.float32)

		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		loss_object = tf.keras.losses.MeanSquaredError()

		for _ in range(self.n_epochs):
		    optimizer.minimize(_loss_gamma, [gamma])
		    self.loss_gamma.append(_loss_gamma().numpy())

		self.gamma = gamma.numpy()

		return self 

	def fit_beta(self):

		def _loss_beta():
		    
		    nu = S + tf.matmul(X, beta)
		    log_likelihood = delta * (tf.math.log(dS) + nu) - tf.exp(nu)

		    return -1.0 * log_likelihood

		beta = tf.Variable(self.beta, dtype=tf.float32)

		X = tf.cast(self.X, dtype=tf.float32)
		S = tf.cast(self.S, dtype=tf.float32)
		dS = tf.cast(self.dS, dtype=tf.float32)
		delta = tf.cast(self.delta, dtype=tf.float32)

		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
	
		for _ in range(self.n_epochs):
		    optimizer.minimize(_loss_beta, [beta])
		    #print("Loss beta:", np.mean(_loss_beta().numpy()))

		self.beta = beta.numpy()

		return self 

	def fit(self):

		self.fit_gamma()

		# update spline matrices 
		self.S = self.Z_long @ self.gamma
		self.dS = self.dZ @ self.gamma[1:]
		
		if np.min(self.dS) < 0:
			assert False, "negative values in dS"

		self.fit_beta()

		return self

	def update_weights(self, gamma, beta):

		self.gamma, self.beta = gamma, beta