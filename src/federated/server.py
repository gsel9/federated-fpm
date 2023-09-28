import numpy as np 
import tensorflow as tf 


class Server:
	
	def __init__(self, clients, epochs, n_client_samples, seed=42):

		self.clients = clients
		self.epochs = epochs
		self.seed = seed

		self.client_weights = n_client_samples / sum(n_client_samples)

		self.gamma, self.beta = None, None 

	# TODO: init gamma params at server 
	def fit_gamma(self):

		for epoch in range(self.epochs):

			for i, client in enumerate(self.clients):
				client.fit_gamma()

			self.gamma, _ = self.aggregate_avg(self.clients)

			for client in self.clients:
				client.update_weights(gamma=self.gamma)

	# TODO: init beta params at server 
	def fit_beta(self):

		for epoch in range(self.epochs):

			for i, client in enumerate(self.clients):
				client.fit_beta()

			_, self.beta = self.aggregate_avg(self.clients)

			for client in self.clients:
				client.update_weights(beta=self.beta)

	def aggregate_avg(self, clients):

		fed_avg = lambda values: np.average(values, axis=0, weights=self.client_weights)

		gammas, betas = [], []
		for client in clients:

			gammas.append(client.gamma)
			betas.append(client.beta)
		
		return fed_avg(gammas), fed_avg(betas)
