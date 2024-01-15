import numpy as np 
import matplotlib.pyplot as plt


def plot_loss(loss, path_to_fig, ref_loss=None, title=None):

	plt.figure(figsize=(5, 4))

	plt.plot(loss, label="Federated")
	if ref_loss is not None:
		plt.plot(ref_loss, label="Centralized")

	if title is not None:
		plt.title(title)

	plt.xlabel("Sever epoch")
	plt.ylabel("Aggregated loss value")

	plt.legend()
	plt.tight_layout()
	plt.savefig(path_to_fig)


def plot_coefficients(coefs, path_to_fig, ref_coefs=None, title=None):

	plt.figure(figsize=(5, 4))

	plt.plot(coefs, label="Federated", marker="o", linestyle="", markersize=8)
	if ref_coefs is not None:
		plt.plot(ref_coefs, label="Centralized", marker="o", linestyle="", markersize=6)

	if title is not None:
		plt.title(title)

	plt.xlabel("Coefficient number")
	plt.xlabel("Coefficient value")

	plt.legend()
	plt.tight_layout()
	plt.savefig(path_to_fig) 