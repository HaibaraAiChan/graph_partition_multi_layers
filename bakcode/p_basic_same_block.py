import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
# from block_dataloader import generate_dataloader
from block_dataloader_graph import generate_dataloader, get_global_graph_edges_ids_2
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
# import deepspeed
import random
from graphsage_model import SAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data
from load_graph import load_ogbn_mag
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean
# from utils import draw_graph_global
from dgl.data.utils import load_graphs
import pickle


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

def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1)==labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device):
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
		pred = model.inference(g, nfeat, device, args)
	model.train()
	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def load_block_subtensor(nfeat, labels, blocks, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	return batch_inputs, batch_labels

def get_mini_batch_size(full_len,num_batch):
	mini_batch=int(full_len/num_batch)
	if full_len%num_batch>0:
		mini_batch+=1
	# print('current mini batch size of output nodes ', mini_batch)
	return mini_batch

def get_total_src_length(blocks):
    res=0
    for block in blocks:
        src_len=len(block.srcdata['_ID'])
        res+=src_len
    return res

#### Entry point
def run(args, device, data):
	# Unpack data
	g, feats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(feats[0])

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])

	full_batch_size = len(train_nid)
	args.batch_size = get_mini_batch_size(full_batch_size,args.num_batch)
	# full_batch_dataloader = dgl.dataloading.NodeDataLoader(
	# 	g,
	# 	train_nid,
	# 	sampler,
	# 	batch_size=full_batch_size,
	# 	shuffle=True,
	# 	drop_last=False,
	# 	num_workers=args.num_workers)
	
	model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)


	epoch_train_CPU_time_list = []
	# see_memory_usage("-----------------------------------------before for epoch loop ")

	for epoch in range(args.num_epochs):
		print('Epoch ' + str(epoch))
		file_name='\\DATA\\re\\fan_out_'+args.fan_out+'\\'+args.dataset+'_'+str(epoch)+'_dataloader_.pickle'
			# Load data (deserialize)
		with open(file_name, 'rb') as handle:
			block_dataloader = pickle.load(handle)
		
	
		print('========after full batch subgraphs of data loading===================================================')
	
		loss_sum =0
		for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
	
			# Load the input features as well as output labels
			batch_inputs, batch_labels = load_block_subtensor(feats, labels, blocks, device)
			blocks = [block.int().to(device) for block in blocks]
			#----------------------------------------------------------------------------------------
			# Compute loss and prediction
			with torch.set_grad_enabled(True):
				# see_memory_usage("----------------------------------------before batch_pred = model(blocks, batch_inputs) ")
				batch_pred = model(blocks, batch_inputs)
				# see_memory_usage("-----------------------------------------batch_pred = model(blocks, batch_inputs) ")
				pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)
				# print('----------------------------------------------------------pseudo_mini_loss ', pseudo_mini_loss)
				pseudo_mini_loss = pseudo_mini_loss
				# print('----------------------------------------------------------pseudo_mini_loss ', pseudo_mini_loss)
				pseudo_mini_loss.backward()
				loss_sum += pseudo_mini_loss

				optimizer.step()
				optimizer.zero_grad()

		# optimizer.step()
		# optimizer.zero_grad()
		print('--------------------------------------------------full batch train loss  ' + str(loss_sum.tolist()))

	print('='*100)
	train_acc = evaluate(model, g, feats, labels, train_nid, device)
	print('train Acc: {:.6f}'.format(train_acc))
	test_acc = evaluate(model, g, feats, labels, test_nid, device)
	print('Test Acc: {:.6f}'.format(test_acc))
	


def main(args):
    
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
	

if __name__=='__main__':
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)

	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--num-runs', type=int, default=2)
	argparser.add_argument('--num-epochs', type=int, default=6)
	argparser.add_argument('--num-hidden', type=int, default=16)

	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='2,2')
	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='20')
	# argparser.add_argument('--fan-out', type=str, default='10,25')
	# argparser.add_argument('--fan-out', type=str, default='2,2,2')
	argparser.add_argument('--num-batch', type=int, default=1)
	argparser.add_argument('--batch-size', type=int, default=24) # karate full batch train output nodes 
	# argparser.add_argument('--batch-size', type=int, default=140) # cora full batch train output nodes 


	# argparser.add_argument('--batch-size', type=int, default=157393)
	# argparser.add_argument('--batch-size', type=int, default=78697)
	# argparser.add_argument('--batch-size', type=int, default=39349)
	# argparser.add_argument('--batch-size', type=int, default=19675)
	# argparser.add_argument('--batch-size', type=int, default=9838)
	# argparser.add_argument('--batch-size', type=int, default=4919)

	# argparser.add_argument('--batch-size', type=int, default=153431)
	# argparser.add_argument('--batch-size', type=int, default=76716)


	# argparser.add_argument('--batch-size', type=int, default=196571)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	# argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)
	

	argparser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")
	argparser.add_argument("--R", type=int, default=5,
                        help="number of hops")

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument('--inductive', action='store_true',
		help="Inductive learning setting") #The store_true option automatically creates a default value of False
	argparser.add_argument('--data-cpu', action='store_true',
		help="By default the script puts all node features and labels "
		     "on GPU when using it to save time for data copy. This may "
		     "be undesired if they cannot fit in GPU memory at once. "
		     "This flag disables that.")
	args = argparser.parse_args()

	set_seed(args)
	
	main(args)
