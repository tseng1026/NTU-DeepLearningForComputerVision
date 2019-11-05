import argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", default="./hw2_data/")
	parser.add_argument("-w", default="./warning.txt")
	parser.add_argument("-m", default="./model_best.pth.tar")
	parser.add_argument("-p", default="./")
	parser.add_argument("--mode", default="baseline")

	args = parser.parse_args()
	return args
