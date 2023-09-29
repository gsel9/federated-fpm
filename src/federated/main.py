import numpy as np 
import matplotlib.pyplot as plt

from sksurv.datasets import load_whas500

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis

from centralised import centralised_benchmark
from splines import NaturalCubicSpline, knots
from client import create_client 
from server import Server 


def data():

	# heart attack survival data
	X, y = load_whas500()
	delta, time = list(zip(*y))

	# retain only numerical covariates 
	X = X.loc[:, ["age", "bmi", "diasbp", "hr", "los", "sysbp"]].values

	#scaler = StandardScaler()
	#X = scaler.fit_transform(X)

	delta = np.array(delta).astype(int)
	logtime = np.log(np.array(time))

	# HACK: otherwise neg values in dS
	to_keep = logtime > 0

	return X[to_keep], y[to_keep][:, None], delta[to_keep][:, None], logtime[to_keep][:, None]


def distributed_data_idx(n_samples, n_clients, seed=42):

	np.random.seed(seed)
	idxs = np.array_split(range(n_samples), n_clients)

	for i, idx in enumerate(idxs):

		idx.sort()
		print(f"N samples client {i + 1}:", len(idx))
	
	return idxs


def eval_spline_experiment(server, clients, delta, logtime):

	plt.figure()
	for i, client in enumerate(clients):
		plt.plot(client.loss_gamma, label=f"C{i+1}")
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
	gamma = server.gamma
	knots_x, knots_y = knots(logtime, delta)

	ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)
	Z = ncs.transform(knots_y, derivative=False)

	plt.figure()
	plt.plot(knots_x, (Z @ gamma).squeeze(), marker="o", linestyle="", label="aggregated")
	plt.plot(knots_x, knots_y, marker="o", linestyle="", label="knots")
	plt.legend()
	plt.savefig("./figures/knots_global.pdf")

	# global logtime 
	plt.figure()
	ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)
	Z = ncs.transform(logtime, derivative=False)
	logtime_hat = (Z @ gamma).squeeze()
	plt.plot(np.linspace(0, 1, len(logtime)), sorted(logtime), label="reference")
	plt.plot(np.linspace(0, 1, len(logtime_hat)), sorted(logtime_hat), label="estimate")
	plt.legend()
	plt.savefig(f"./figures/logtime.pdf")


def eval_beta_experiment(server, clients, X, y, delta, logtime):

	plt.figure()
	for i, client in enumerate(clients):
		plt.plot(client.loss_beta, label=f"C{i+1}")
	plt.legend()
	plt.savefig("./figures/loss_beta.pdf")

	beta_star = server.beta

	# baseline predictor comparison
	model = LogisticRegression()
	model.fit(X, delta)

	y_pred_lr = model.predict(X)
	cmat_lr = confusion_matrix(delta, y_pred_lr)

	model = CoxPHSurvivalAnalysis()
	model.fit(X, y)

	y_pred_ph = (model.predict(X).squeeze() > 0).astype(int)
	cmat_ph = confusion_matrix(delta, y_pred_ph)

	y_pred_fl = ((X @ beta_star).squeeze() > 0).astype(int)
	cmat_fl = confusion_matrix(delta, y_pred_fl)

	fig, axes = plt.subplots(ncols=3, figsize=(12, 6))
	axes[0].set_title(label="LR")
	axes[1].set_title(label="PH")
	axes[2].set_title(label="FL")

	display = ConfusionMatrixDisplay(cmat_lr)
	display.plot(ax=axes[0], colorbar=False)
	display = ConfusionMatrixDisplay(cmat_ph)
	display.plot(ax=axes[1], colorbar=False)
	display = ConfusionMatrixDisplay(cmat_fl)
	display.plot(ax=axes[2], colorbar=False)

	plt.savefig("./figures/cmats_beta.pdf")


def eval_centralised_benchmark(X, y, delta, beta, gamma, logtime, loss_gamma, loss_beta):

	plt.figure()
	plt.plot(loss_gamma)
	plt.savefig("./figures/loss_gamma_centralised.pdf")

	plt.figure()
	plt.plot(loss_beta)
	plt.savefig("./figures/loss_beta_centralised.pdf")

	knots_x, knots_y = knots(logtime, delta)	

	# global logtime 
	plt.figure()
	ncs = NaturalCubicSpline(knots=knots_x, order=1, intercept=True)
	Z = ncs.transform(logtime, derivative=False)
	logtime_hat = (Z @ gamma).squeeze()
	plt.plot(np.linspace(0, 1, len(logtime)), sorted(logtime), label="reference")
	plt.plot(np.linspace(0, 1, len(logtime_hat)), sorted(logtime_hat), label="estimate")
	plt.legend()
	plt.savefig(f"./figures/logtime_centralised.pdf")

	# baseline predictor comparison
	model = LogisticRegression()
	model.fit(X, delta)

	y_pred_lr = model.predict(X)
	cmat_lr = confusion_matrix(delta, y_pred_lr)

	model = CoxPHSurvivalAnalysis()
	model.fit(X, y)
	
	y_pred_ph = (model.predict(X).squeeze() > 0).astype(int)
	cmat_ph = confusion_matrix(delta, y_pred_ph)

	y_pred_fl = ((X @ beta).squeeze() > 0).astype(int)
	cmat_fl = confusion_matrix(delta, y_pred_fl)

	fig, axes = plt.subplots(ncols=3, figsize=(12, 6))
	axes[0].set_title("LR")
	axes[1].set_title("PH")
	axes[2].set_title("FL")

	display = ConfusionMatrixDisplay(cmat_lr)
	display.plot(ax=axes[0], colorbar=False)
	display = ConfusionMatrixDisplay(cmat_ph)
	display.plot(ax=axes[1], colorbar=False)
	display = ConfusionMatrixDisplay(cmat_fl)
	display.plot(ax=axes[2], colorbar=False)

	plt.savefig("./figures/cmats_beta_centralised.pdf")


def initialize_parameters(size_beta, size_gamma, seed=42):

	np.random.seed(42)
	beta = np.random.normal(size=(size_beta, 1))
	gamma = np.random.normal(size=(size_gamma, 1))

	return gamma, beta 


def main():
	# TODO: wrap in flower 

	n_clients = 1

	order = 1
	intercept = True

	n_knots = 6

	learning_rate = 0.05 
	local_epochs = 1 #40
	global_epochs = 200 #5 

	X, y, delta, logtime = data()

	gamma_init, beta_init = initialize_parameters(size_beta=X.shape[1], 
												  size_gamma=int(intercept) + order + n_knots - 1) 
	# distributed data 
	idxs = distributed_data_idx(X.shape[0], n_clients)

	# TODOOO: Check get same losses

	clients = []
	for i, idx in enumerate(idxs):
		clients.append(create_client(X[idx], delta[idx], logtime[idx], local_epochs, learning_rate, n_knots))
		print(f"Created client: {i+1}")
		
	server = Server(clients, global_epochs, learning_rate, gamma_init, beta_init)
	#server.fit_gamma()
	#server.fit_beta()
	server.fit_gamma_gradients()
	server.fit_beta_gradients()
	
	#eval_spline_experiment(server, clients, delta, logtime)
	#eval_beta_experiment(server, clients, X, y, delta, logtime)
	
	gamma, beta, loss_gamma, loss_beta = centralised_benchmark(X[idxs[0]], delta[idxs[0]], logtime[idxs[0]], n_knots, gamma_init, beta_init,
	##gamma, beta, loss_gamma, loss_beta = centralised_benchmark(X, delta, logtime, n_knots, gamma_init, beta_init,
															   learning_rate, global_epochs, order, intercept)
	#eval_centralised_benchmark(X, y, delta, beta, gamma, logtime, loss_gamma, loss_beta)

	#np.save("./results/beta_central.npy", beta)
	#np.save("./results/gamma_central.npy", gamma)

	#beta = np.load("./results/beta_central.npy")
	#gamma = np.load("./results/gamma_central.npy")
	print(np.linalg.norm(server.gamma - gamma))
	print(np.linalg.norm(server.beta - beta))


if __name__ == "__main__":
	main()