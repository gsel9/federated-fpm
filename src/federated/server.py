import numpy as np 
import tensorflow as tf 


class Server:
	
	def __init__(self, clients, epochs, seed=42):

		self.clients = clients
		self.epochs = epochs
		self.seed = seed

		self.gamma, self.beta = None, None 

	def fit_gamma(self):

		for epoch in range(self.epochs):

			for i, client in enumerate(self.clients):
				client.fit_gamma()

			self.gamma, _ = self.aggregate_avg(self.clients)

			for client in self.clients:
				client.update_weights(gamma=self.gamma)

	def update_client_splines(self):

		for client in self.clients:
			client.update_splines()

	def fit_beta(self):

		for epoch in range(self.epochs):

			for i, client in enumerate(self.clients):
				client.fit_beta()

			_, self.beta = self.aggregate_avg(self.clients)

			for client in self.clients:
				client.update_weights(beta=self.beta)

	def fit_gamma_gradients(self):

		self.optimizer_gamma = tf.keras.optimizers.Adam(learning_rate=0.05)
		gamma_variable = tf.Variable(self.clients[0].gamma)

		for epoch in range(self.epochs):

			gradients = 0
			for i, client in enumerate(self.clients):
				gradients += client.gamma_gradients(gamma_variable)

			self.optimizer_gamma.apply_gradients([(gradients, gamma_variable)])

		self.gamma = gamma_variable.numpy()

		for client in self.clients:
			client.update_weights(gamma=self.gamma)

	def fit_beta_gradients(self, epochs=100):

		beta_variable = tf.Variable(self.clients[0].beta)

		for epoch in range(epochs):

			dl_dbs, d2l_db2s = 0, 0
			for i, client in enumerate(self.clients):
				
				dl_db, d2l_db2 = client.beta_gradients(beta_variable)
				dl_dbs += dl_db
				d2l_db2s += d2l_db2

			beta_variable = beta_variable - dl_dbs / d2l_db2s

		self.beta = beta_variable.numpy()

	@staticmethod
	def fed_sum(weights):
		return np.sum(weights, axis=0)

	def aggregate_avg(self, clients):

		agg_average = lambda values: np.mean(values, axis=0)

		gammas, betas = [], []
		for client in clients:

			gammas.append(client.gamma)
			betas.append(client.beta)
		
		return agg_average(gammas), agg_average(betas)
