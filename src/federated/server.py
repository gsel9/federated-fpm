import numpy as np 


class Server:
	
	def __init__(self, clients, epochs, seed=42):

		self.clients = clients
		self.epochs = epochs
		self.seed = seed

		self.gamma, self.beta = None, None 

		# TEMP 
		optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

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
		# TODO: sum the likelihood gradients and update the weights 
		for epoch in range(self.epochs):

			for i, client in enumerate(self.clients):
				client.fit_beta()

			_, self.beta = self.aggregate_gradients(self.clients)
			#_, self.beta = self.aggregate_avg(self.clients)

			for client in self.clients:
				client.update_weights(beta=self.beta)

	@staticmethod
	def fed_sum(weights):
		return np.sum(weights, axis=0)

	@staticmethod
	def fed_avg(weights):
		return np.mean(weights, axis=0)

	def aggregate_gradients(self, clients):

		gammas, betas = [], []

		for client in clients:

			gammas.append(client.gamma)
			betas.append(client.beta)

		# TODO: 
		# - sum gradients 
		# - sever does one step of gradient descent 
		
		return self.fed_sum(gammas), self.fed_sum(betas)
	
	def aggregate_avg(self, clients):

		gammas, betas = [], []

		for client in clients:

			gammas.append(client.gamma)
			betas.append(client.beta)
		
		return self.fed_avg(gammas), self.fed_avg(betas)