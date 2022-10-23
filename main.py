from data import load_dev_data
from utils import results_to_disk 
from experiments import train_eval 


def main():

	data = load_dev_data()

	results = train_eval(data, algorithm)

	results_to_disk(results, "results/dev", format="csv")


if __name__ == "__main__":
	main()