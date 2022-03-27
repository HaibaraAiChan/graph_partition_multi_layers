import torch
import dgl
import numpy
import time
from itertools import islice
from statistics import mean
from multiprocessing import Manager, Pool
from multiprocessing import Process, Value, Array
from graph_partitioner import Graph_Partitioner
from my_utils import gen_batch_output_list
from draw_graph import draw_graph, draw_dataloader_blocks_pyvis,draw_dataloader_blocks_pyvis_total
from draw_nx import draw_nx_graph
def unique_tensor_item(combined):
	uniques, counts = combined.unique(return_counts=True)
	return uniques.type(torch.long)


def unique_edges(edges_list):
	temp = []
	for i in range(len(edges_list)):
		tt = edges_list[i]  # tt : [[],[]]
		for j in range(len(tt[0])):
			cur = (tt[0][j], tt[1][j])
			if cur not in temp:
				temp.append(cur)
	# print(temp)   # [(),(),()...]
	res_ = list(map(list, zip(*temp)))  # [],[]
	res = tuple(sub for sub in res_)
	return res


def generate_random_mini_batch_seeds_list(OUTPUT_NID, args):
	'''
	Parameters
	----------
	OUTPUT_NID: final layer output nodes id (tensor)
	args : all given parameters collection

	Returns
	-------
	'''
	selection_method = args.selection_method
	mini_batch = args.batch_size
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	if selection_method == 'random':
		indices = torch.randperm(full_len)  # get a permutation of the index of output nid tensor (permutation of 0~n-1)
	else: #selection_method == 'range'
		indices = torch.tensor(range(full_len))

	output_num = len(OUTPUT_NID.tolist())
	map_output_list = list(numpy.array(OUTPUT_NID)[indices.tolist()])
	batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
	weights_list = []
	for i in batches_nid_list:
		temp = len(i)/output_num
		weights_list.append(len(i)/output_num)
		
	return batches_nid_list, weights_list


def get_global_graph_edges_ids_2(raw_graph, block_to_graph):
    
	edges=block_to_graph.edges(order='eid')
	edge_src_local = edges[0]
	edge_dst_local = edges[1]
	induced_src = block_to_graph.srcdata[dgl.NID]
	induced_dst = block_to_graph.dstdata[dgl.NID]
		
	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
	# raw_src = block_to_graph.ndata[dgl.NID]['_N_src'][src] 
	# raw_dst= block_to_graph.ndata[dgl.NID]['_N_dst'][dst]
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)


def get_global_graph_edges_ids(raw_graph, cur_block):
	'''
		Parameters
		----------
		raw_graph : graph
		cur_block: (local nids, local nids): (tensor,tensor)

		Returns
		-------
		global_graph_edges_ids: []                    current block edges global id list
	'''

	src, dst = cur_block.all_edges(order='eid')
	src = src.long()
	dst = dst.long()
	# print(src.tolist())
	# print(dst.tolist())
	raw_src, raw_dst = cur_block.srcdata[dgl.NID][src], cur_block.dstdata[dgl.NID][dst]
	# print(raw_src.tolist())
	# print(raw_dst.tolist())
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)


def generate_one_block(raw_graph, global_eids, global_srcnid, global_dstnid):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	_graph = dgl.edge_subgraph(raw_graph, global_eids,store_ids=True)
	edge_src_list = _graph.edges(order='eid')[0].tolist()
	edge_dst_list = _graph.edges(order='eid')[1].tolist()
	eid_list = _graph.edata['_ID'].tolist()
	print(sorted(eid_list))
	dst_local_nid_list=[]
	[dst_local_nid_list.append(nid) for nid in edge_dst_list if nid not in dst_local_nid_list]
	# to keep the order of dst nodes
	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))

	new_block.dstdata[dgl.NID] = global_dstnid
	new_block.srcdata[dgl.NID] = global_srcnid
	print(new_block.edata['_ID'].tolist())
	# new_block.edata['_ID']=_graph.edata['_ID']
	# local_eids=[i for i in range(len(global_eids))]
	# new_block.edata['_ID']=torch.tensor(local_eids,dtype=torch.long)
	print()
	print('*************-------------------new_block------------------')
	print(new_block)
	print(new_block.edges()[0])
	print(new_block.edges()[1])
	print(new_block.edges(form='all')[2])
	print(new_block.srcdata)
	print(new_block.dstdata)

	return new_block

def check_connections_0(batched_nodes_list, current_layer_subgraph):
	res=[]
	induced_src = current_layer_subgraph.srcdata[dgl.NID]
	induced_dst = current_layer_subgraph.dstdata[dgl.NID]
	eids_global = current_layer_subgraph.edata['_ID']
	print('current layer subgraph eid (global)')
	
	print(sorted(eids_global.tolist()))
	src_nid_list = induced_src.tolist()
	# multi-layers model: current_layer_subgraph, here
	# print('\n *************************************   src nid of current layer subgraph')
	print(src_nid_list)
	src_local, dst_local, index = current_layer_subgraph.edges(form='all')
	print( current_layer_subgraph.edges(form='all')[0])
	print( current_layer_subgraph.edges(form='all')[1])
	print( current_layer_subgraph.edges(form='all')[2])
	# src_local, dst_local = current_layer_subgraph.edges(order='eid')

	src, dst = induced_src[src_local], induced_src[dst_local]

	dict_nid_2_local = {src_nid_list[i]: i for i in range(0, len(src_nid_list))}
	
	src_compare=[]
	dst_compare=[]
	compare=[]
	prev_eids=[]
	for step, output_nid in enumerate(batched_nodes_list):
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		
		local_in_edges_tensor = current_layer_subgraph.in_edges(local_output_nid, form='all')
		# return (𝑈,𝑉,𝐸𝐼𝐷)
		# get local srcnid and dstnid from subgraph
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
		
		mini_batch_dst_local= list(local_in_edges_tensor)[1]
		mini_batch_dst_global= induced_src[mini_batch_dst_local].tolist()
		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
			print('local dst not match')
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		
		add_src=[i for i in mini_batch_src_global if i not in output_nid]
		r_ = []
		[r_.append(x) for x in add_src if x not in r_]
		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)
		
		res.append((src_nid, output_nid, global_eid_tensor))
		# res.append((src_nid, output_nid, global_eid_tensor, mini_batch_src_global, mini_batch_dst_global))
		compare.append(global_eid_tensor.tolist())
		src_compare.append(src_nid.tolist())
		dst_compare.append(output_nid.tolist())

	tttt=sum(compare,[])
	print(sorted(list(set(tttt))))
	if set(tttt)!= set(eids_global.tolist()):
		print('the edges not match')
		print(sorted(list(set(tttt))))
		print(sorted(list(set(eids_global.tolist()))))
	if set(sum(src_compare,[]))!= set(induced_src.tolist()):
		print('the src nodes not match')
		print(set(sum(src_compare,[])))
		print(set(induced_src.tolist()))
	if set(sum(dst_compare,[]))!= set(induced_dst.tolist()):
		print('the dst nodes not match')
		print(set(sum(dst_compare,[])))
		print(set(induced_dst.tolist()))
	return res


def generate_blocks_for_one_layer(raw_graph, block_2_graph, batches_nid_list):
	
	layer_src = block_2_graph.srcdata[dgl.NID]
	layer_dst = block_2_graph.dstdata[dgl.NID]
	layer_eid = block_2_graph.edata[dgl.NID].tolist()
	print(sorted(layer_eid))
	
	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_0(batches_nid_list, block_2_graph)
	t2 = time.time()
	check_connection_time.append(t2-t1) #------------------------------------------
	src_list=[]
	dst_list=[]
	ll=len(batches_temp_res_list)

	src_compare=[]
	dst_compare=[]
	eid_compare=[]
	for step, (srcnid, dstnid, current_block_global_eid) in enumerate(batches_temp_res_list):
	# for step, (srcnid, dstnid, current_block_global_eid, src_e, dst_e) in enumerate(batches_temp_res_list):
		# print('batch ' + str(step) + '-' * 30)
		t_ = time.time()
		if step == ll-1:
			print()
		# if len(prev_batched_eid_list) and prev_batched_eid_list[step]:
		# 	new_eids=current_block_global_eid.tolist()
		# 	pure_new_eid=[]
		# 	[pure_new_eid.append(eid) for eid in new_eids if eid not in pure_new_eid and eid not in prev_batched_eid_list[step]] #remove duplicate
		# 	current_block_global_eid=prev_batched_eid_list[step]+pure_new_eid
		# 	current_block_global_eid=torch.tensor(current_block_global_eid, dtype=torch.long)
		# 	cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)
		# else:
		# 	cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)
		cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)

		t__=time.time()
		block_generation_time.append(t__-t_)  #------------------------------------------
		#----------------------------------------------------
		print('batch: ', step)
		induced_src = cur_block.srcdata[dgl.NID]
		induced_dst = cur_block.dstdata[dgl.NID]
		induced_eid = cur_block.edata[dgl.NID].tolist()
		print('src and dst nids')
		print(induced_src)
		print(induced_dst)
		e_src_local, e_dst_local = cur_block.edges(order='eid')
		e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
		e_src = e_src.detach().numpy().astype(int)
		e_dst = e_dst.detach().numpy().astype(int)

		combination = [p for p in zip(e_src, e_dst)]
		print('batch block graph edges: ')
		print(combination)
		#----------------------------------------------------
		blocks.append(cur_block)
		src_list.append(srcnid)
		dst_list.append(dstnid)

		eid_compare.append(induced_eid)
		src_compare.append(induced_src.tolist())
		dst_compare.append(induced_dst.tolist())

	tttt=sum(eid_compare,[])
	print((set(tttt)))
	if set(tttt)!= set(layer_eid):
		print('the edges not match')
		print(sorted(list((set(tttt)))))
		print(sorted(list(set(layer_eid))))
	if set(sum(src_compare,[]))!= set(layer_src.tolist()):
		print('the src nodes not match')
		print(set(sum(src_compare,[])))
		print(set(layer_src.tolist()))
	if set(sum(dst_compare,[]))!= set(layer_dst.tolist()):
		print('the dst nodes not match')
		print(set(sum(dst_compare,[])))
		print(set(layer_dst.tolist()))

		# data_loader.append((srcnid, dstnid, [cur_block]))
		
	# print("\nconnection checking time " + str(sum(check_connection_time)))
	# print("total of block generation time " + str(sum(block_generation_time)))
	# print("average of block generation time " + str(mean(block_generation_time)))
	connection_time = sum(check_connection_time)
	block_gen_time = sum(block_generation_time)
	mean_block_gen_time = mean(block_generation_time)


	return blocks, src_list,dst_list,(connection_time, block_gen_time, mean_block_gen_time)



def generate_dataloader_w_partition(raw_graph, block_to_graph_list, args):
	for layer, block_to_graph in enumerate(block_to_graph_list):
		
		current_block_eidx, current_block_edges = get_global_graph_edges_ids_2(raw_graph, block_to_graph)
		block_to_graph.edata['_ID'] = current_block_eidx
		if layer == 0:
			my_graph_partitioner=Graph_Partitioner(block_to_graph, args) #init a graph partitioner object
			batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list=my_graph_partitioner.init_graph_partition()

			print('partition_len_list')
			print(p_len_list)
			args.batch_size=my_graph_partitioner.batch_size
			
			blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph, batched_output_nid_list)
			# TODO
			#change the generate block
			connection_time, block_gen_time, mean_block_gen_time = time_1
			# batch_list_generation_time = t1 - tt
			time_2 = (connection_time, block_gen_time, mean_block_gen_time, batch_list_generation_time)
		else:
			return
	data_loader=[]
	# TODO
	return data_loader, weights_list, time_2

def gen_grouped_dst_list(prev_layer_blocks):
	post_dst=[]
	for block in prev_layer_blocks:
		src_nids = block.srcdata['_ID'].tolist()
		post_dst.append(src_nids)
	return post_dst # return next layer's dst nids(equals prev layer src nids)

def generate_dataloader_wo_gp_Pure_range(raw_graph, block_to_graph_list, args):
	data_loader=[]
	weights_list=[]
	num_batch=0
	blocks_list=[]
	final_dst_list =[]
	final_src_list=[]
	prev_layer_blocks=[]
	t_2_list=[]
	# prev_layer_src_list=[]
	# prev_layer_dst_list=[]
	print('now we generate block from output to src direction, bottom up direction')
	l=len(block_to_graph_list)
	# the order of block_to_graph_list is bottom-up(the smallest block at first order)
	#b it means the graph partition starts 
	# from the output layer to the first layer input block graphs.
	for layer, block_to_graph in enumerate(block_to_graph_list):
		dst_nids=block_to_graph.dstdata['_ID'].tolist()
		print('The real block id is ', l-1-layer)
		print('dst nids ', sorted(dst_nids))
		src_nids=block_to_graph.srcdata['_ID'].tolist()
		# print('block ', layer)
		print('src nids ', sorted(src_nids))
		# print(src_nids)
		current_block_eidx, current_block_edges = get_global_graph_edges_ids_2(raw_graph, block_to_graph)
		# block_to_graph.edata['_ID'] = current_block_eidx  # raw global edata
		# print(' layer edges')
		# print(sorted(current_block_eidx.tolist()))

		if layer ==0:
			print(current_block_eidx)
			print(current_block_edges[0].tolist())
			print(current_block_edges[1].tolist())

			# src_nids=block_to_graph.srcdata['_ID'].tolist()
			# print('time of batches_nid_list generation : ' + str(t1 - tt) + ' sec')
			t1=time.time()
			indices = [i for i in range(len(dst_nids))]
			batched_output_nid_list, w_list=gen_batch_output_list(dst_nids,indices,args.batch_size)
			tt=time.time()
			weights_list=w_list
			num_batch=len(batched_output_nid_list)
			print('num of batch ',num_batch )
			print('layer ', layer)
			print('\tselection method range initialization spend ', time.time()-t1)
			# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
			blocks, src_list, dst_list,time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph,  batched_output_nid_list)
			connection_time, block_gen_time, mean_block_gen_time = time_1
			batch_list_generation_time = tt - t1
			time_2 = [connection_time, block_gen_time, mean_block_gen_time, batch_list_generation_time]
			t_2_list.append(time_2)
			prev_layer_blocks=blocks
			# prev_layer_dst_list=dst_list
			# prev_layer_src_list=src_list

			blocks_list.append(blocks)
			final_dst_list=dst_list
			if layer==args.num_layers-1:
				final_src_list=src_list
			# final_src_list=src_list

		else:
			t1=time.time()
			
			grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
			num_batch=len(grouped_output_nid_list)
			print('num of batch ',num_batch )
			
			tt=time.time()
			print('layer ',layer)
			print('\tselection method range initialization spend ', time.time()-t1)
			
			blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph, grouped_output_nid_list)
			connection_time, block_gen_time, mean_block_gen_time = time_1
			batch_list_generation_time = tt-t1
			time_2 = [connection_time, block_gen_time, mean_block_gen_time, batch_list_generation_time]
			t_2_list.append(time_2)

			# final_dst_list=grouped_output_nid_list

			if layer==args.num_layers-1: # if current block is the final block, the src list will be the final src
				final_src_list=src_list
				# print()
			else:
				prev_layer_blocks=blocks
    
			# blocks_list.insert(blocks,0)
			blocks_list.append(blocks)

	
	for batch_id in range(num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][batch_id])
		# print('batch ', batch_id)
		# print(cur_blocks)
		# print()
		
		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks))
	# return


	sum_list=[]
	if len(t_2_list)==1:
		sum_list=t_2_list[0]
	elif len(t_2_list)==2:
		for bb in range(0,len(t_2_list),2):
			list1=t_2_list[bb]
			list2=t_2_list[bb+1]
			for (item1, item2) in zip(list1, list2):
				sum_list.append(item1+item2)

	elif len(t_2_list)==3:
		for bb in range(0,len(t_2_list),3):
			list1=t_2_list[bb]
			list2=t_2_list[bb+1]
			list3=t_2_list[bb+2]
			for (item1, item2, item3) in zip(list1, list2, list3):
				sum_list.append(item1+item2+item3)

	return data_loader, weights_list, sum_list
		

def generate_dataloader(raw_graph, block_to_graph_list, args):
    
    
    if 'partition' in args.selection_method:
        return generate_dataloader_w_partition(raw_graph, block_to_graph_list, args)
    else:
        return generate_dataloader_wo_gp_Pure_range(raw_graph, block_to_graph_list, args)
		# return generate_dataloader_0(raw_graph, block_to_graph, args)
