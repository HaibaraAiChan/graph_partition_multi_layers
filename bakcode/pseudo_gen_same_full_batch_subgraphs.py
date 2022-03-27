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
from graphsage_model import SAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data
from load_graph import load_ogbn_mag
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory
import pickle
# from utils import draw_graph_global
from dgl.data.utils import save_graphs


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

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels


#### Entry point
def run(args, device, data):
		# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in_feats--------------------------------------')
	print(in_feats)
	# dataloader_device = torch.device('cpu')

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])

	full_batch_size = len(train_nid)
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		sampler,
		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	
	model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	
	avg_src=[]
	import numpy
	while True:
		avg_src=[]
		for epoch in range(args.num_epochs):
			print('Epoch ' + str(epoch))
			for full_batch_step, (input_nodes, output_seeds, full_batch_blocks) in enumerate(full_batch_dataloader):
				# print('full_batch_blocks')
				print(full_batch_blocks)
				l=len(full_batch_blocks)
				avg_src.append(len(input_nodes))
				for layer, cur_block in enumerate(full_batch_blocks):
					block_to_graph=dgl.block_to_graph(cur_block)
					block_to_graph.srcdata['_ID']=cur_block.srcdata['_ID']
					block_to_graph.dstdata['_ID']=cur_block.dstdata['_ID']
					block_to_graph.edata['_ID']=cur_block.edata['_ID']
					save_graphs('./DATA/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_Block_'+str(layer)+'_subgraph.bin',[block_to_graph])
		

				batch_inputs, batch_labels = load_subtensor(nfeats, labels, output_seeds, input_nodes, device)
				full_batch_blocks = [block.int().to(device) for block in full_batch_blocks]
				batch_pred = model(full_batch_blocks, batch_inputs)
				loss = loss_fcn(batch_pred, batch_labels)
				print('-----------------------------------full batch loss ' + str(loss.tolist()))
		
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
		

				
		print('mean src ', numpy.mean(avg_src))
		
		if args.fan_out=='2,2,2' or args.fan_out=='2,3,3':
			if args.dataset=='karate' and abs((numpy.mean(avg_src) - 34) )< 4:
				return
		
		if args.fan_out=='1,1':
			if args.dataset=='karate' and abs((numpy.mean(avg_src) - 29)) < 2:     ### 1,1
				return

		if args.fan_out=='2,2' or args.fan_out=='2,3':
			if args.dataset=='karate' and abs((numpy.mean(avg_src) - 31) )< 4:     ### 2,2
				return
		
		#-------------------simplified------------------------------------
		# if args.fan_out=='2,2' or args.fan_out=='2,3':
		# 	if args.dataset=='karate' and abs((numpy.mean(avg_src) - 12) )< 4:     ### 2,2
		# 		return
		# if args.fan_out=='2,2,2' or args.fan_out=='2,2,3':
		# 	if args.dataset=='karate' and abs((numpy.mean(avg_src) - 16) )< 4:     ### 2,2
		# 		return
		#-------------------------------------------------------	
		if args.fan_out=='10,25':
			if args.dataset=='karate' and abs((numpy.mean(avg_src) - 34) )< 2:     ### 10,25
				return
			if args.dataset=='cora' and abs((numpy.mean(avg_src) - 1352)) < 5:     ### 10,25
				return
			if args.dataset=='reddit'  and abs((numpy.mean(avg_src) - 227878)) < 10:   ### 10, 25 reddit dataset
				return
		
		if args.fan_out=='10':
			if args.dataset=='karate' and abs((numpy.mean(avg_src) - 31) )< 2:     ### 10
				return
			if args.dataset=='cora' and abs((numpy.mean(avg_src) - 585)) < 5:     ### 10
				return
			if args.dataset=='reddit'  and abs((numpy.mean(avg_src) - 217248)) < 10:   ### 10 reddit dataset
				return
		
		# if args.dataset=='reddit' and args.fan_out =='100' and abs((numpy.mean(avg_src) - 226365)) < int(args.fan_out):   ### reddit dataset
		# 	return
		if args.fan_out=='2':
			if args.dataset=='karate' and abs((numpy.mean(avg_src) - 30) )< 5:
				return
		
			


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
	# argparser.add_argument('--fan-out', type=str, default='2,3')
	# argparser.add_argument('--fan-out', type=str, default='10,25')
	argparser.add_argument('--fan-out', type=str, default='2,2')
	# argparser.add_argument('--fan-out', type=str, default='1,1')

	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='2,2,2')
	# argparser.add_argument('--fan-out', type=str, default='2,3,3')

	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='2')


	argparser.add_argument('--batch-size', type=int, default=157393)


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