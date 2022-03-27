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
from gat_model_full_arxiv import  GAT

import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed
from load_graph import load_ogbn_mag,load_ogbn_dataset
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
	if args.gpu >= 0:
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
		pred = model(g, nfeats)
		# pred = model.inference(g, nfeat, device, args)
	model.train()
	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


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

# def accuracy(logits, labels):
#     _, indices = torch.max(logits, dim=1)
#     correct = torch.sum(indices == labels)
#     return correct.item() * 1.0 / len(labels)

# def evaluate(model, features, labels, mask):
#     model.eval()
#     with torch.no_grad():
#         logits = model(features)
#         logits = logits[mask]
#         labels = labels[mask]
#         return accuracy(logits, labels)

def train(model, g, feats, y_true, train_idx, optimizer):
	model.train()

	optimizer.zero_grad()
	see_memory_usage("----------------------------------------before batch_pred = model ")
	out = model(g, feats)[train_idx]
	see_memory_usage("----------------------------------------after batch_pred = model ")
	loss = F.nll_loss(out, y_true[train_idx])
	loss.backward()
	optimizer.step()
	see_memory_usage("----------------------------------------- after optimizer.step() ")
	return loss.item()

# @torch.no_grad()
# def test(model, g, feats, y_true, split_idx, evaluator):
# 	# y_true=torch.tensor([y_true.tolist()])
# 	model.eval()

# 	out = model(g, feats)
# 	# y_pred = out.argmax(dim=-1, keepdim=True)
# 	y_pred = out.argmax(dim=-1)

# 	train_acc = evaluator.eval({
# 		'y_true': y_true[split_idx['train']],
# 		'y_pred': y_pred[split_idx['train']],
# 	})['acc']
# 	valid_acc = evaluator.eval({
# 		'y_true': y_true[split_idx['valid']],
# 		'y_pred': y_pred[split_idx['valid']],
# 	})['acc']
# 	test_acc = evaluator.eval({
# 		'y_true': y_true[split_idx['test']],
# 		'y_pred': y_pred[split_idx['test']],
# 	})['acc']

# 	return train_acc, valid_acc, test_acc



def run(args, device, data):
		# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid, split_idx = data
	in_feats = len(nfeats[0])
	print('in_feats--------------------------------------')
	print(in_feats)
	
	full_batch_size = len(train_nid)
	args.batch_size = full_batch_size

	# g=g.to(device)
	g = g.int().to(device)
	nfeats=nfeats.to(device)
	labels=labels.to(device)
	train_nid = train_nid.to(device)
	val_nid = val_nid.to(device)
	test_nid = test_nid.to(device)
	model = GAT(num_layers=args.num_layers,
				in_feats=nfeats.size(-1),
				num_hidden=args.num_hidden,
				num_classes=n_classes,
				heads=[4, 4, 4],
				feat_drop=args.dropout,
				attn_drop=args.dropout).to(device)
	
	model = model.to(device)
	
	logger = Logger(args.num_runs, args)

	dur = []
	for run in range(args.num_runs):
		model.reset_parameters()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		for epoch in range(1, 1 + args.num_epochs):
			
			t0 = time.time()
			loss = train(model, g, nfeats, labels, train_nid, optimizer)
			if epoch >= 3:
				dur.append(time.time() - t0)
				print('Training time/epoch {}'.format(np.mean(dur)))

			if not args.eval:
				continue
			result = evaluate(model, g, nfeats, labels, split_idx['train'],split_idx['valid'],split_idx['test'],device)
			# result = test(model, g, nfeats, labels, split_idx, evaluator)
			logger.add_result(run, result)

			if epoch % args.log_steps == 0:
				train_acc, valid_acc, test_acc = result
				print(f'Run: {run + 1:02d}, '
					f'Epoch: {epoch:02d}, '
					f'Loss: {loss:.4f}, '
					  f'Train: {100 * train_acc:.2f}%, '
					  f'Valid: {100 * valid_acc:.2f}% '
					  f'Test: {100 * test_acc:.2f}%')
		if args.eval:
			logger.print_statistics(run)
	if args.eval:
		logger.print_statistics()


def load_train(args):
	device = "cpu"
	if args.dataset=='ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		
	else:
		raise Exception('unknown dataset')
	device = "cuda:0"
	
	best_test = run(args, device, data)
	


def main():
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--root', type=str, default='../benchmark_full_graph/dataset/')
	argparser.add_argument('--log_steps', type=int, default=1)

	argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument("--num-layers", type=int, default=3,
						help="number of hidden layers")
	argparser.add_argument("--lr", type=float, default=0.0029739421726400865,
						help="learning rate")
	argparser.add_argument('--weight-decay', type=float, default=2.4222556964495987e-05,
						help="weight decay")
	argparser.add_argument("--num-hidden", type=int, default=16,
						help="number of hidden units")
	argparser.add_argument("--dropout", type=float, default=0.18074706609292976,
						help="Dropout to use")
	
	
	argparser.add_argument("--eval", action='store_true',default=True,
						help='If not set, we will only do the training part.')
	argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--num-runs', type=int, default=10)
	argparser.add_argument('--num-epochs', type=int, default=500)

	argparser.add_argument("---batch-size", type=int, default=0,
						help="batch size")
	argparser.add_argument("--R", type=int, default=5,
						help="number of hops")
	

	args = argparser.parse_args()

	set_seed(args)
	
	load_train(args)



if __name__=='__main__':
	main()
	