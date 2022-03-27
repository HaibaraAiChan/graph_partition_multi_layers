import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from block_dataloader import generate_dataloader
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
# import deepspeed
import random
from gat_model_full_reddit import  GAT
# from graphsage_model import SAGE, GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed
from load_graph import load_ogbn_mag
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory
import pickle

from utils import Logger
import os 
import numpy

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

# def calc_acc(logits, labels, mask):
#     logits = logits[mask]
# 	labels = labels[mask]
# 	_, indices = torch.max(logits, dim=1)
# 	correct = torch.sum(indices == labels)
# 	return correct.item() * 1.0 / len(labels)

def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	model.eval()
	with torch.no_grad():
		# pred = model(blocks=None, x=nfeats, g=g)
		pred = model(nfeats)
		# pred = model.inference(g, nfeat, device, args)
	model.train()
	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)

# def calc_acc(logits, labels, mask):
#     logits = logits[mask]
#     labels = labels[mask]
#     _, indices = torch.max(logits, dim=1)
#     correct = torch.sum(indices == labels)
#     return correct.item() * 1.0 / len(labels)

# def evaluate(model, features, labels, train_mask, val_mask, test_mask):
#     model.eval()
#     with torch.no_grad():
#         logits = model(features)
#         train_acc = calc_acc(logits, labels, train_mask)
#         val_acc = calc_acc(logits, labels, val_mask)
#         test_acc = calc_acc(logits, labels, test_mask)
#         return train_acc, val_acc, test_acc

# def evaluate(model, g, nfeat, labels, val_nid, device):
# 	"""
# 	Evaluate the model on the validation set specified by ``val_nid``.
# 	g : The entire graph.
# 	inputs : The features of all the nodes.
# 	labels : The labels of all the nodes.
# 	val_nid : the node Ids for validation.
# 	device : The GPU device to evaluate on.
# 	"""
# 	model.eval()
# 	with torch.no_grad():
# 		pred = model.inference(g, nfeat, device, args)
# 	model.train()
# 	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
#### Entry point


def run(args, device, data):
		# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in_feats--------------------------------------')
	print(in_feats)
	
	full_batch_size = len(train_nid)
	args.batch_size = full_batch_size

	# g=g.to(device)
	g = dgl.add_self_loop(g)
	g = g.int().to(device)
	nfeats=nfeats.to(device)
	labels=labels.to(device)
	train_nid = train_nid.to(device)
	val_nid = val_nid.to(device)
	test_nid = test_nid.to(device)

	# model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model = GAT(g=g,
                num_layers=args.num_layers,
                in_feats=in_feats,
                num_hidden=args.num_hidden,
                num_classes=n_classes,
                heads=[1, 1, 1],
                feat_drop=args.dropout,
                attn_drop=args.dropout)
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()

	logger = Logger(args.num_runs, args)
	dur = []
	for run in range(args.num_runs):
		model.reset_parameters()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		for epoch in range(args.num_epochs):
				
			model.train()
			if epoch >= 3:
				t0 = time.time()
			# forward
			see_memory_usage("----------------------------------------before batch_pred = model ")
			batch_pred = model(nfeats)
			see_memory_usage("-----------------------------------------batch_pred = model ")
			# loss = loss_fcn(batch_pred, labels)
			loss = loss_fcn(batch_pred[train_nid], labels[train_nid])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			see_memory_usage("----------------------------------------- after optimizer.step() ")
				
			if epoch >= 3:
				dur.append(time.time() - t0)
				print('Training time/epoch {}'.format(np.mean(dur)))

			if not args.eval:
				continue

			train_acc, val_acc, test_acc = evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device)
			logger.add_result(run, (train_acc, val_acc, test_acc))

			print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss.item(), train_acc, val_acc, test_acc))

		if args.eval:
			logger.print_statistics(run)

	if args.eval:
		logger.print_statistics()
		
		

def load_train(args):
	device = "cpu"
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')
	
	best_test = run(args, device, data)
	


def main():
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--dataset', type=str, default='cora')
	argparser.add_argument("--device", type=int, default=0)
	argparser.add_argument('--seed', type=int, default=1238)
	argparser.add_argument("--num-layers", type=int, default=3,
						help="number of hidden layers")
	argparser.add_argument("--lr", type=float, default=0.005,
						help="learning rate")
	argparser.add_argument('--weight-decay', type=float, default=5e-4,
						help="weight decay")
	argparser.add_argument("--num-hidden", type=int, default=8,
						help="number of hidden units")
	argparser.add_argument("--dropout", type=float, default=.6,
						help="Dropout to use")
	argparser.add_argument('--num-epochs', type=int, default=200)
	argparser.add_argument('--num-runs', type=int, default=10)
	
	
	argparser.add_argument("--eval", action='store_true', default=True,
						help='If not set, we will only do the training part.')

	argparser.add_argument("---batch-size", type=int, default=0,
						help="batch size")
	argparser.add_argument("--R", type=int, default=5,
						help="number of hops")

	args = argparser.parse_args()

	set_seed(args)
	
	load_train(args)


if __name__=='__main__':
	main()
	