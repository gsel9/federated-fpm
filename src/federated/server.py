import numpy as np 
import tensorflow as tf 


class Server:
	
	def __init__(self, clients, epochs, learning_rate, gamma, beta, seed=42):

		self.clients = clients
		self.epochs = epochs
		self.seed = seed

		self.learning_rate = learning_rate
		self.gamma = gamma 
		self.beta = beta 

		self._distribute_client_params(gamma, beta)
		self.weights = self._request_client_weights()

	def _distribute_client_params(self, gamma=None, beta=None):

		for client in self.clients:
			client.update_weights(gamma=gamma, beta=beta)

	def _request_client_weights(self):

		n_client_samples = []

		for client in self.clients:
			n_client_samples.append(client.n_samples)

		return np.array(n_client_samples) / sum(n_client_samples)

	def request_spline_update(self):

		for client in self.clients:
			client.update_splines()

	def fit_gamma_gradients(self):

		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		gamma_variable = tf.Variable(self.gamma, dtype=tf.float32)
		
		for epoch in range(self.epochs):

			gradients = 0
			for i, client in enumerate(self.clients):

				#gradients += self.weights[i] * client.gamma_gradients()
				gradients += client.gamma_gradients()
			
			optimizer.apply_gradients([(gradients, gamma_variable)])
			
			self._distribute_client_params(gamma=gamma_variable.numpy())
		self.gamma = gamma_variable.numpy()

	def fit_beta_gradients(self):
		
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		beta_variable = tf.Variable(self.beta, dtype=tf.float32)

		for epoch in range(self.epochs):

			gradients = 0 
			for i, client in enumerate(self.clients):
				gradients += client.beta_gradients()

			optimizer.apply_gradients([(gradients, beta_variable)])
		
			self._distribute_client_params(beta=beta_variable.numpy())
		self.beta = beta_variable.numpy() 
