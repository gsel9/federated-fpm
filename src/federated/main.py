import numpy as np 

from sksurv.datasets import load_whas500

#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#from sksurv.linear_model import CoxPHSurvivalAnalysis

from centralised import centralised_benchmark
from evaluation import plot_loss, plot_coefficients
from splines import NaturalCubicSpline, knots
from client import create_client 
from server import Server 


def data():

	# heart attack survival data
	X, y = load_whas500()
	delta, time = list(zip(*y))

	# retain only numerical covariates 
	X = X.loc[:, ["age", "bmi", "diasbp", "hr", "los", "sysbp"]].values

	# NOTE: assume no need for local standardisation 
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	delta = np.array(delta).astype(int)
	logtime = np.log(np.array(time))

	# HACK: otherwise neg values in dS
	to_keep = logtime > 0

	return X[to_keep], y[to_keep][:, None], delta[to_keep][:, None], logtime[to_keep][:, None]


def distributed_data_idx(n_samples, n_clients, stratifier, seed=42):

	idxs = np.array_split(range(n_samples), n_clients)

	for i, idx in enumerate(idxs):

		idx.sort()
		print(f"N samples client {i + 1}:", len(idx))
	
	return idxs


def initialize_parameters(size_beta, size_gamma, seed=42):

	np.random.seed(42)
	beta = np.random.normal(size=(size_beta, 1))
	gamma = np.random.normal(size=(size_gamma, 1))

	return gamma, beta 


def main():
	# NOTE:
	# - goal: assess if federated flexible parametric models can be used with FL 
	# - assuming iid data and shared knot statistics the federated flexible parametric models equals 
	#   the centralised model 
	# - extensions baseline hazard models 
	#	- basic polynomial regression
	#	- kernel smoother (local polynomial regression)
	# - extensions federated analytics 
	#	- standard error, z-statistics and p-values 
	# - extenstions realistic federated settings 
	#	- non-iid data (data shifts, VI days)
	#   - local iterations to boost optimisation
	#	- privacy 

	n_clients = 3

	order = 1
	intercept = True

	n_knots = 6

	learning_rate = 0.05 
	local_epochs = 1
	global_epochs = 50  

	X, y, delta, logtime = data()

	# ERROR: size_gamma is prolly wrong 
	gamma_init, beta_init = initialize_parameters(size_beta=X.shape[1], 
												  size_gamma=int(intercept) + order + n_knots - 1) 
	# distributed data 
	idxs = distributed_data_idx(X.shape[0], n_clients, delta)

	# assume global knots for now 
	knots_x, knots_y = knots(logtime, delta, n_knots)

	clients = []
	for i, idx in enumerate(idxs):
		clients.append(create_client(X[idx], delta[idx], logtime[idx], local_epochs, learning_rate, knots_x, knots_y, n_knots))
		print(f"Created client: {i+1}")
	
	server = Server(clients, global_epochs, learning_rate, gamma_init, beta_init)

	server.fit_gamma_gradients()
	server.request_spline_update()
	server.fit_beta_gradients()
	
	gamma, beta, loss_gamma, loss_beta = centralised_benchmark(X, delta, logtime, n_knots, gamma_init, beta_init, knots_x, knots_y,
															   learning_rate, global_epochs, order, intercept)
	
	plot_loss(server.loss_gamma, "figures/loss_gamma.pdf", ref_loss=loss_gamma, title="loss gamma")
	plot_loss(server.loss_beta, "figures/loss_beta.pdf", ref_loss=loss_beta, title="loss beta")

	plot_coefficients(server.gamma, "figures/coefs_gamma.pdf", ref_coefs=gamma, title="coefs gamma")
	plot_coefficients(server.beta, "figures/coefs_beta.pdf", ref_coefs=beta, title="coefs beta")

	#np.save("./results/beta_central.npy", beta)
	#np.save("./results/gamma_central.npy", gamma)

	#beta = np.load("./results/beta_central.npy")
	#gamma = np.load("./results/gamma_central.npy")

	print("gamma diff:", np.linalg.norm(server.gamma - gamma))
	print("beta diff:", np.linalg.norm(server.beta - beta))


if __name__ == "__main__":
	main()