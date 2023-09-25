import numpy as np 


class Server:
	
	def __init__(self, seed=42):

		self.seed = seed

	def fed_avg(self, weights):

		return np.mean(weights, axis=0)

	def aggregate(self, clients):

		gammas, betas = [], []

		for client in clients:

			gammas.append(client.gamma)
			betas.append(client.beta)
		
		weights_avg = [self.fed_avg(gammas), 
					   self.fed_avg(betas)]
		
		return weights_avg