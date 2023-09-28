import numpy as np 
import tensorflow as tf 


class Server:
	
	def __init__(self, clients, epochs, seed=42):

		self.clients = clients
		self.epochs = epochs
		self.seed = seed

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

	# TODO: sample number weighting coefficients 
	def aggregate_avg(self, clients):

		agg_average = lambda values: np.mean(values, axis=0)

		gammas, betas = [], []
		for client in clients:

			gammas.append(client.gamma)
			betas.append(client.beta)
		
		return agg_average(gammas), agg_average(betas)
