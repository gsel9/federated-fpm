import numpy as np 
#import pandas as pd
#import tensorflow as tf

import matplotlib.pyplot as plt

from sksurv.datasets import load_whas500
#from scipy.interpolate import CubicSpline

#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.metrics import confusion_matrix
#from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

#from sksurv.linear_model import CoxPHSurvivalAnalysis
from splines import NaturalCubicSpline, knots
from client import create_client 
from server import Server 


def data():

	# heart attack survival data
	X, y = load_whas500()
	delta, time = list(zip(*y))

	# retain only numerical covariates 
	X = X.loc[:, ["age", "bmi", "diasbp", "hr", "los", "sysbp"]].values

	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	delta = np.array(delta).astype(int)
	logtime = np.log(np.array(time))

	# HACK
	to_keep = logtime > 0

	return X[to_keep], delta[to_keep], logtime[to_keep]


def distributed_data_idx(n_samples, n_clients, seed=42):

	np.random.seed(seed)
	idxs = np.array_split(range(n_samples), n_clients)

	for i, idx in enumerate(idxs):

		idx.sort()
		print(f"N samples client {i + 1}:", len(idx))

	return idxs 


def spline_experiment(server, clients, epochs):

	for _ in range(epochs):
		for i, client in enumerate(clients):
	
			client.fit_gamma()

		gamma, beta = server.aggregate(clients)

		for client in clients:
			client.update_weights(gamma, beta)


def eval_spline_experiment(server, clients, delta, logtime):

	plt.figure()
	plt.plot(clients[0].loss_gamma, label="C3")
	plt.plot(clients[1].loss_gamma, label="C3")
	plt.plot(clients[2].loss_gamma, label="C3")
	plt.legend()
	plt.savefig("./figures/loss_gamma.pdf")

	# local knots 
	plt.figure()
	for i, client in enumerate(clients):
	
		knots_x, knots_y = knots(client.logtime, client.delta)
		ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)
		Z = ncs.transform(knots_y, derivative=False)

		plt.plot(knots_x, knots_y, marker="o", linestyle="", label=f"knots C{i+1}")
		plt.plot(knots_x, (Z @ client.gamma).squeeze(), marker="o", linestyle="", label=f"estimate C{i+1}")

	plt.legend()
	plt.savefig(f"./figures/knots_local.pdf")

	# global knots 
	gamma, _ = server.aggregate(clients)
	knots_x, knots_y = knots(logtime, delta)

	ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)
	Z = ncs.transform(knots_y, derivative=False)

	plt.figure()
	plt.plot(knots_x, (Z @ gamma).squeeze(), marker="o", linestyle="", label="aggregated")
	plt.plot(knots_x, knots_y, marker="o", linestyle="", label="knots")
	plt.legend()
	plt.savefig("./figures/knots_global.pdf")


def main():

	learning_rate = 0.05

	local_epochs = 40
	global_epochs = 5

	X, delta, logtime = data()

	# distributed data 
	idxs = distributed_data_idx(X.shape[0], 3)

	server = Server()

	clients = [
		create_client(X[idxs[0]], delta[idxs[0]], logtime[idxs[0]], local_epochs, learning_rate),
		create_client(X[idxs[1]], delta[idxs[1]], logtime[idxs[1]], local_epochs, learning_rate),
		create_client(X[idxs[2]], delta[idxs[2]], logtime[idxs[2]], local_epochs, learning_rate)
	]

	# NOTE: key is running enough local epochs (40 local and 5 global)
	#spline_experiment(server, clients, global_epochs)
	#eval_spline_experiment(server, clients, delta, logtime)

	#gamma_star, _ = server.aggregate(clients)
	#np.save(gamma_star, "./results/gamma_star.npy")
	
	#beta_experiment()
	#eval_beta_experiment()

	# TODO: 
	# - plot individual loss values per client (single plot)
	# - tune training parameters 
	# - convergence criteria on gamma and beta? 

	#np.save(beta_star, "beta_star.npy")

	# baseline copmarison
	#y_pred = ((X @ beta_star).squeeze() > 0).astype(int)
	#confusion_matrix(delta, y_pred)
	#model = CoxPHSurvivalAnalysis()
	#model.fit(X, y)
	#y_pred = (model.predict(X).squeeze() > 0).astype(int)
	#confusion_matrix(delta, y_pred)


if __name__ == "__main__":
	main()