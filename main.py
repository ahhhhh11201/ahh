import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf', default="./utils/conf.json")
	args = parser.parse_args()

	with open(args.conf, 'r') as f:
		conf = json.load(f)

	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	server = Server(conf, eval_datasets)
	clients = []
	att_client = [2,5]
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))

	print("\n\n")
	for e in range(conf["global_epochs"]):

		candidates = random.sample(clients, conf["k"])

		weight_accumulator = {}

		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		# num_c=0
		for c in candidates:

			diff = c.local_train(server.global_model)

			for name, params in server.global_model.state_dict().items():
				# if num_c in att_client:
				# 	diff[name] *= -4
				weight_accumulator[name].add_(diff[name])
			# num_c += 1

		server.model_aggregate(weight_accumulator)

		acc, loss = server.model_eval()

		# print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))







