import dgl
import torch as th
# from torch._C import int64
import torch.nn as nn
import dgl.function as fn
import tqdm
import dgl.nn.pytorch as dglnn
from memory_usage import see_memory_usage
import torch.nn.functional as F
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import SAGEConv

# class SAGEConv(nn.Module):
# 	def __init__(self,
# 				 in_feats,
# 				 out_feats,
# 				 aggre):
# 		super(SAGEConv, self).__init__()

# 		self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
# 		self._out_feats = out_feats
# 		self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
# 		self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)
# 		self.aggre = aggre
# 		self.reset_parameters()

# 	def reset_parameters(self):
# 		"""Reinitialize learnable parameters."""
# 		gain = nn.init.calculate_gain('relu')
# 		nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
# 		nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

# 	def forward(self, graph, feat):
# 		r"""Compute GraphSAGE layer.

# 		Parameters
# 		----------
# 		graph : DGLGraph
# 			The graph.
# 		feat : torch.Tensor or pair of torch.Tensor
# 			If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
# 			:math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
# 			If a pair of torch.Tensor is given, the pair must contain two tensors of shape
# 			:math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

# 		Returns
# 		-------
# 		torch.Tensor
# 			The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
# 			is size of output feature.
# 		"""
# 		graph = graph.local_var()

# 		if isinstance(feat, tuple):
# 			feat_src, feat_dst = feat
# 		else:
# 			feat_src = feat_dst = feat

# 		h_self = feat_dst

# 		graph.srcdata['h'] = feat_src
# 		if self.aggre == 'mean':
# 			graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
# 		h_neigh = graph.dstdata['neigh']
# 		rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

# 		return rst

class GraphSAGE(nn.Module):
	def __init__(self,
				in_feats,
				hidden_feats,
				out_feats,
				num_layers,
				aggre,
				activation,
				dropout):
		super(GraphSAGE, self).__init__()

		self.layers = nn.ModuleList()
		self.bns = nn.ModuleList()
		# input layer
		self.layers.append(SAGEConv(in_feats, hidden_feats, aggre))
		# hidden layers
		for _ in range(num_layers - 2):
			self.layers.append(SAGEConv(hidden_feats, hidden_feats, aggre))
		# output layer
		self.layers.append(SAGEConv(hidden_feats, out_feats, aggre))
		self.dropout = nn.Dropout(p=dropout)
		self.activation=activation

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()

	def forward(self, g, x):
		for i, layer in enumerate(self.layers[:-1]):
			x = layer(g, x)
			x = self.activation(x)
			x = self.dropout(x)
		x = self.layers[-1](g, x)

		return x.log_softmax(dim=-1)

# class SAGEConv(nn.Module):
# 	def __init__(self,
# 				in_feats,
# 				out_feats,
# 				aggr,
# 				feat_drop=0.,
# 				activation=None):
# 		super(SAGEConv, self).__init__()

# 		self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
# 		self._out_feats = out_feats
# 		self._aggr = aggr
# 		self.feat_drop = nn.Dropout(feat_drop)
# 		self.activation = activation
# 		self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
# 		self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)
# 		self.reset_parameters()

# 	def reset_parameters(self):
# 		"""Reinitialize learnable parameters."""
# 		gain = nn.init.calculate_gain('relu')
# 		nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
# 		nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

# 	def forward(self, graph, feat):
# 		r"""Compute GraphSAGE layer.

# 		Parameters
# 		----------
# 		graph : DGLGraph
# 			The graph.
# 		feat : torch.Tensor or pair of torch.Tensor
# 			If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
# 			:math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
# 			If a pair of torch.Tensor is given, the pair must contain two tensors of shape
# 			:math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

# 		Returns
# 		-------
# 		torch.Tensor
# 			The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
# 			is size of output feature.
# 		"""
# 		graph = graph.local_var()

# 		if isinstance(feat, tuple):
# 			feat_src = self.feat_drop(feat[0])
# 			feat_dst = self.feat_drop(feat[1])
# 		else:
# 			feat_src = feat_dst = self.feat_drop(feat)

# 		h_self = feat_dst

# 		graph.srcdata['h'] = feat_src
# 		if self._aggr == 'sum':
# 			graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
# 		elif self._aggr == 'mean':
# 			graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
# 		else:
# 			return ValueError("Expect aggregation to be 'sum' or 'mean', got {}".format(self._aggr))
# 		h_neigh = graph.dstdata['neigh']
# 		rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

# 		# activation
# 		if self.activation is not None:
# 			rst = self.activation(rst)
# 		return rst


# class GraphSAGE(nn.Module):
# 	def __init__(self,
# 				in_feats,
# 				n_hidden,
# 				n_classes,
# 				aggr,
# 				activation=F.relu,
# 				dropout=0.):
# 		super(GraphSAGE, self).__init__()
		
# 		self.layers = nn.ModuleList()
# 		# self.g = g
# 		self.layers.append(SAGEConv(in_feats, n_hidden, aggr, activation=activation))
# 		self.layers.append(SAGEConv(n_hidden, n_classes, aggr, feat_drop=dropout, activation=None))

# 	def reset_parameters(self):
# 		for layer in self.layers:
# 			layer.reset_parameters()
	
# 	def forward(self, g, x):
# 		h = x
# 		for layer in self.layers:
# 			h = layer(g, h)
# 		return h




# class SAGE(nn.Module):
# 	def __init__(self,
# 				in_feats,
# 				n_hidden,
# 				n_classes,
# 				n_layers,
# 				activation,
# 				dropout,
# 				aggre):
# 		super().__init__()
# 		self.n_layers = n_layers
# 		self.n_hidden = n_hidden
# 		self.n_classes = n_classes
# 		self.aggre = aggre
# 		self.layers = nn.ModuleList()
		
		
# 		self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, self.aggre, activation=activation))
# 		self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, self.aggre, feat_drop=dropout, activation=None))
		
# 		# self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, self.aggre))
# 		# self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, self.aggre))
# 		self.dropout = nn.Dropout(dropout)
# 		self.activation = activation
		
# 		# if n_layers == 1:
# 		# 	self.layers.append(dglnn.SAGEConv(in_feats, n_classes, self.aggre))
# 		# if n_layers >= 2:
# 		# 	self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, self.aggre))
# 		# 	for i in range(0, n_layers - 2):
# 		# 		self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, self.aggre))
# 		# 	self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, self.aggre))
# 		# self.dropout = nn.Dropout(dropout)
# 		# self.activation = activation

# 	def reset_parameters(self):
# 		for layer in self.layers:
# 			layer.reset_parameters()

# 	def forward(self, g, x):
# 		h = x
# 		for layer in self.layers:
# 			h = layer(g, h)
# 		return h

# 	# def forward(self, g, x):  
# 	# 	h = x
# 	# 	for l_id, layer in enumerate(self.layers):
# 	# 		h = layer(g, h)
# 	# 		# if l_id != len(self.layers) - 1:
# 	# 		# 	h = self.activation(h)
# 	# 		# if l_id == len(self.layers) - 1: # the same with benchmark's configuration
# 	# 		# 	h = self.dropout(h)
			
# 	# 	return h




# 	# def forward(self, blocks, x):
# 	# 	h = x
# 	# 	for l, (layer, block) in enumerate(zip(self.layers, blocks)):
# 	# 		h = layer(block, h)
# 	# 		if l!=len(self.layers) - 1:
# 	# 			h = self.activation(h)
# 	# 			h = self.dropout(h)
# 	# 	return h

# 	# def forward(self, blocks, x, g=None):  # compatible with full graph and blocks
# 	# 	if not blocks:
# 	# 		h = x
# 	# 		for l, layer in enumerate(self.layers):
# 	# 			h = layer(g, h)
				
# 	# 			if l != len(self.layers) - 1:
# 	# 				h = self.activation(h)
# 	# 			if l == len(self.layers) - 1: # the same with benchmark's configuration
# 	# 				h = self.dropout(h)

# 	# 		return h
# 	# 	h = x
# 	# 	# print('x')
# 	# 	# print(x)
# 	# 	for l, (layer, block) in enumerate(zip(self.layers, blocks)):
# 	# 		# print('sage model to process train ----------&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&S')
# 	# 		# see_memory_usage('before layer (block, h)')
# 	# 		h = layer(block, h)
		
# 	# 		if l != len(self.layers) - 1:
# 	# 			h = self.activation(h)
# 	# 		if l == len(self.layers) - 1: # the same with benchmark's configuration
# 	# 			h = self.dropout(h)
# 	# 		# if l != len(self.layers) - 1:
# 	# 		# 	h = self.activation(h)
# 	# 		# 	h = self.dropout(h)

# 	# 	return h

# 	def inference(self, g, x, device, args):
# 		"""
# 		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
# 		g : the entire graph.
# 		x : the input of entire node set.

# 		The inference code is written in a fashion that it could handle any number of nodes and
# 		layers.
# 		"""
# 		# During inference with sampling, multi-layer blocks are very inefficient because
# 		# lots of computations in the first few layers are repeated.
# 		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
# 		# on each layer are of course splitted in batches.
# 		# TODO: can we standardize this?
# 		for l, layer in enumerate(self.layers):
# 			y = th.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)

# 			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
# 			dataloader = dgl.dataloading.NodeDataLoader(
# 				g,
# 				th.arange(g.num_nodes(),dtype=th.long).to(device),
# 				sampler,
# 				# batch_size=24,
# 				batch_size=args.batch_size,
# 				shuffle=True,
# 				drop_last=False,
# 				num_workers=args.num_workers)


# 			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
# 				block = blocks[0]
# 				block = block.to(device)
# 				h = x[input_nodes].to(device)
# 				h = layer(block, h)
# 				if l != len(self.layers) - 1:
# 					h = self.activation(h)
# 				if l == len(self.layers) - 1: # the same with benchmark's configuration
# 					h = self.dropout(h)

# 				y[output_nodes] = h.cpu()

# 			x = y
# 		return y

