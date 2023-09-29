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

		self._distribute_client_params(gamma, beta, update_splines=False)
		self.weights = self._request_client_weights()

	def _distribute_client_params(self, gamma=None, beta=None, update_splines=False):

		for client in self.clients:
			client.update_weights(gamma=gamma, beta=beta, update_splines=update_splines)

	def _request_client_weights(self):

		n_client_samples = []
		for client in self.clients:
			n_client_samples.append(client.n_samples)

		return np.array(n_client_samples) / sum(n_client_samples)

	def _request_spline_update(self):

		for client in self.clients:
			client.update_splines()

	def fit_gamma_gradients(self):

		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		gamma_variable = tf.Variable(self.gamma, dtype=tf.float32)
		
		for epoch in range(self.epochs):

			gradients = []
			for i, client in enumerate(self.clients):
				gradients.append(client.gamma_gradients())
			
			optimizer.apply_gradients([(np.vstack(gradients), gamma_variable)])

			self._distribute_client_params(gamma=gamma_variable.numpy())
		self._request_spline_update()

		self.gamma = gamma_variable.numpy()

	def fit_beta_gradients(self):
		
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		beta_variable = tf.Variable(self.beta, dtype=tf.float32)
		
		for epoch in range(self.epochs):

			gradients = []
			for i, client in enumerate(self.clients):
				gradients.append(client.beta_gradients())

			optimizer.apply_gradients([(np.vstack(gradients), beta_variable)])
			
			self._distribute_client_params(beta=beta_variable.numpy())
		self.beta = beta_variable.numpy() 

	# TODO: init gamma params at server 
	def fit_gamma(self):

		for epoch in range(self.epochs):

			for i, client in enumerate(self.clients):
				client.fit_gamma()

			self.gamma, _ = self.aggregate_avg(self.clients)

		for client in self.clients:
			client.update_weights(gamma=self.gamma, update_splines=True)

	# TODO: init beta params at server 
	def fit_beta(self):

		for epoch in range(self.epochs):

			for i, client in enumerate(self.clients):
				client.fit_beta()

			_, self.beta = self.aggregate_avg(self.clients)

			for client in self.clients:
				client.update_weights(beta=self.beta)

	# TODO: sample number weighting coefficients 
	def aggregate_avg(self, clients):

		agg_average = lambda values: np.mean(values, axis=0)

		gammas, betas = [], []
		for client in clients:

			gammas.append(client.gamma)
			betas.append(client.beta)
		
		return agg_average(gammas), agg_average(betas)
