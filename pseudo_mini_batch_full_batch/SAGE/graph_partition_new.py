import numpy 
import dgl
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean
from my_utils import *
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
# sys.path.insert(0,'..')
from draw_graph import draw_dataloader_blocks_pyvis_total, gen_pyvis_graph_local

class Graph_Partitioner:
	def __init__(self, layer_block, args):
		self.balanced_init_ratio=args.balanced_init_ratio
		self.dataset=args.dataset
		self.layer_block=layer_block # local graph with global nodes indices
		self.local=False
		self.output_nids=layer_block.dstdata['_ID'] # tensor type
		self.local_output_nids=[]
		self.local_src_nids=[]
		self.src_nids_list= layer_block.srcdata['_ID'].tolist()
		self.full_src_len=len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list=[]
		self.local_batched_seeds_list=[]
		self.weights_list=[]
		self.alpha=args.alpha 
		self.walkterm=args.walkterm
		self.num_batch=args.num_batch
		self.selection_method=args.selection_method
		self.batch_size=0
		self.ideal_partition_size=0

		self.bit_dict={}
		self.side=0
		self.partition_nodes_list=[]
		self.partition_len_list=[]

		self.time_dict={}
		self.red_before=[]
		self.red_after=[]
		self.args=args
		
		
	def gen_batched_seeds_list(self):
		'''
		Parameters
		----------
		OUTPUT_NID: final layer output nodes id (tensor)
		selection_method: the graph partition method

		Returns
		-------
		'''
	
		full_len = len(self.local_output_nids)  # get the total number of output nodes
		self.batch_size=get_mini_batch_size(full_len,self.num_batch)
		
		indices=[]
		if self.selection_method == 'range_init_graph_partition' :
			t=time.time()
			indices = [i for i in range(full_len)]
			batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
			print('range_init for graph_partition spend: ', time.time()-t)
		elif self.selection_method == 'random_init_graph_partition' :
			t=time.time()
			indices = random_shuffle(full_len)
			batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
			print('random_init for graph_partition spend: ', time.time()-t)
		elif self.selection_method == 'balanced_init_graph_partition' :
			t=time.time()
			batches_nid_list, weights_list=self.balanced_init()
			print('balanced_init for graph_partition spend: ', time.time()-t)

			
		else:# selection_method == 'shared_neighbor_graph_partition':
			indices = torch.tensor(range(full_len)) #----------------------------TO DO
		
		# batches_nid_list, weights_list=gen_batch_output_list(self.output_nids,indices,self.batch_size)

		self.local_batched_seeds_list=batches_nid_list
		self.weights_list=weights_list

		print('The batched output nid list before graph partition')
		# print_len_of_batched_seeds_list(batches_nid_list)

		return 



	def remove_un_output_nodes(self):
		import copy
		local_src=copy.deepcopy(self.local_src_nids)
		mask_array = np.full(len(local_src),True, dtype=np.bool)
		mask_array[self.local_output_nids] = False
		from itertools import compress
		to_remove=list(compress(local_src, mask_array))
		return to_remove
  
  
  

	def weighted_graph_bipart(self):
		
		u, v =self.layer_block.edges()[0], self.layer_block.edges()[1] # local edges
		g = dgl.graph((u,v))
		nx_g = dgl.to_networkx(g)  # local graph
		A = nx.to_scipy_sparse_matrix(nx_g)
		AT= A.transpose()
		shared_neighbor_matrix = (AT*A).todok()
		shared_neighbor_matrix.setdiag(0)

		
		# shared_neighbor_matrix.eliminate_zeros()
		# tt=set(u.tolist()+v.tolist())
		# remove = [node for node in tt if node not in self.local_output_nids]
		remove = self.remove_un_output_nodes() # use mask to replace the for loop to speed up
		print('remove')
		print(remove)

		print(shared_neighbor_matrix)

		# now we got shared neighbor numbers
		# then we construct auxiliary graph on output nodes for output nodes (batches) partition
		# {(a, b): 2} means 'a and b share two neighbors'
		temp = shared_neighbor_matrix.keys() # [(a,b), (c,d)...]
		
		v=shared_neighbor_matrix.values()
		k=shared_neighbor_matrix.keys()
		nodes_pairs = list(shared_neighbor_matrix.keys())
		output_shared_matrix = {key: value  for key,value in zip(k,v) if (key[1] not in remove )and (key[0] not in remove) }
		#output_shared_matrix=shared_neighbor_matrix.fromkeys(nodes_pairs)
		capacity = list(output_shared_matrix.values())
		o_nodes_pairs = list(output_shared_matrix.keys())
		start=(list(zip(*o_nodes_pairs))[0])
		end=(list(zip(*o_nodes_pairs))[1])
		df=pd.DataFrame({'source':start,'target':end, 'weight': capacity,'capacity':capacity})	
		# df['src']=start
		# df['dst']=end
		# df['capacity']=capacity
		print(df)
		# df=pd.DataFrame((start, end, capacity),columns=['src','dst','capacity'])
		G=nx.from_pandas_edgelist(df,edge_attr=True)
		print(G)
		all_nodes=set(start+end)
		length=len(set(start+end))
		x=start[int(length/2)]
		x=min(min(start),min(end))
		y=max(max(start),max(end))
		
		cut_value, partition = nx.minimum_cut(G, x, y)
		print(cut_value)
		print(partition)
		part1=list(partition[0])
		part2=list(partition[1])

		self.local_batched_seeds_list=[part1,part2]
		
		return 

 
 
	def get_src_len(self,seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src_len= len(list(set(in_ids+seeds)))
		return src_len



	def get_partition_src_len_list(self):
		partition_src_len_list=[]
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))
		
		self.partition_src_len_list=partition_src_len_list
		return partition_src_len_list


	def graph_partition(self):
		
		full_batch_subgraph=self.layer_block #heterogeneous graph (block)
		self.bit_dict={}
		print('----------------------------  graph partition start---------------------')
		# if num_batch == 0:
		#     self.output_nids/
		self.ideal_partition_size=(self.full_src_len/self.num_batch)
		if self.num_batch==2:
			self.weighted_graph_bipart()
		# src_ids=list(full_batch_subgraph.edges())[0]
		# dst_ids=list(full_batch_subgraph.edges())[1]
		# local_g = dgl.graph((src_ids, dst_ids)) #homogeneous graph
		# local_g = dgl.remove_self_loop(local_g)
		# # from draw_graph import draw_graph
		# # draw_graph(local_g)
		# self.layer_block=local_g
		# global block_to_graph
		# block_to_graph=local_g

		# self.gen_batched_seeds_list() # based on user choice to split output nodes 

		# src_len_list= self.get_partition_src_len_list()


		weight_list=get_weight_list(self.local_batched_seeds_list)
		src_len_list=self.get_partition_src_len_list()
		print('after graph partition')
		self.weights_list=weight_list
		self.partition_len_list=src_len_list

		return self.local_batched_seeds_list, weight_list, src_len_list
	
	def global_to_local(self):
		
		sub_in_nids = self.src_nids_list
		print('src global')
		print(sub_in_nids)
		global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
		self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))
		print('dst local')
		print(self.local_output_nids)
		self.local_src_nids = list(map(global_nid_2_local.get, self.src_nids_list))
		
		self.local=True
		return 	


	def local_to_global(self):
		sub_in_nids = self.src_nids_list
		local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
		
		global_batched_seeds_list=[]
		for local_in_nids in self.local_batched_seeds_list:
			global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
			global_batched_seeds_list.append(global_in_nids)

		self.global_batched_seeds_list=global_batched_seeds_list
		print('-----------------------------------------------global batched output nodes id----------------------------')
		for inp in self.global_batched_seeds_list:
			
			print(sorted(inp))
		# print(self.global_batched_seeds_list)
		self.local=False
		return 


	def init_graph_partition(self):
		ts = time.time()
		
		self.global_to_local() # global to local            self.local_batched_seeds_list
		print('global_2_local spend time (sec)', (time.time()-ts))
		
		# t1 = time.time()
		# self.gen_batched_seeds_list()

		t2=time.time()
		# Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
		self.graph_partition()
		print('graph partition algorithm spend time', time.time()-t2)
		# after that, we transfer the nids of batched output nodes from local to global.
		self.local_to_global() # local to global         self.global_batched_seeds_list
		t_total=time.time()-ts

		return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list
