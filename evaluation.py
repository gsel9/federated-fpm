import numpy as np 
import matplotlib.pyplot as plt


def plot_loss(loss, path_to_fig, ref_loss=None, title=None):

	plt.figure(figsize=(8, 6))

	plt.plot(loss, label="loss")
	if ref_loss is not None:
		plt.plot(ref_loss, label="reference loss")

	if title is not None:
		plt.title(title)

	plt.xlabel("Sever epoch")

	plt.legend()
	plt.tight_layout()
	plt.savefig(path_to_fig)


def plot_coefficients(coefs, path_to_fig, ref_coefs=None, title=None):

	plt.figure(figsize=(8, 6))

	plt.plot(coefs, label="coefs", marker="o", linestyle="", markersize=8)
	if ref_coefs is not None:
		plt.plot(ref_coefs, label="reference coefs", marker="o", linestyle="", markersize=6)

	if title is not None:
		plt.title(title)

	plt.xlabel("Coefficient number")

	plt.legend()
	plt.tight_layout()
	plt.savefig(path_to_fig) 