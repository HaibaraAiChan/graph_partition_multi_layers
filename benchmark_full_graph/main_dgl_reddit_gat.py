import argparse
import dgl
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import load_data
from dgl.nn.pytorch import GATConv

from utils import Logger


import random
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

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_feats,
                 num_hidden,
                 num_classes,
                 heads,
                 activation=F.elu,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.g = g
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_feats=in_feats,
                                       out_feats=num_hidden,
                                       num_heads=heads[0],
                                       feat_drop=0.,
                                       attn_drop=0.,
                                       negative_slope=negative_slope,
                                       activation=activation))
        # hidden layers
        for l in range(num_layers - 2):
            # due to multi-head, the in_feats = num_hidden * num_heads
            self.gat_layers.append(GATConv(in_feats=num_hidden * heads[l],
                                           out_feats=num_hidden,
                                           num_heads=heads[l + 1],
                                           feat_drop=feat_drop,
                                           attn_drop=attn_drop,
                                           negative_slope=negative_slope,
                                           activation=activation))
        # output projection
        self.gat_layers.append(GATConv(in_feats=num_hidden * heads[-2],
                                       out_feats=num_classes,
                                       num_heads=heads[-1],
                                       feat_drop=feat_drop,
                                       attn_drop=attn_drop,
                                       negative_slope=negative_slope,
                                       activation=None))

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

    def forward(self, h):
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

def calc_acc(logits, labels, mask):
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, train_mask, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        train_acc = calc_acc(logits, labels, train_mask)
        val_acc = calc_acc(logits, labels, val_mask)
        test_acc = calc_acc(logits, labels, test_mask)
        return train_acc, val_acc, test_acc

def main():    
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset", type=str, default='reddit')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--seed', type=int, default=1236)
    parser.add_argument("--num-layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--lr", type=float, default=0.0029739421726400865,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=2.4222556964495987e-05,
                        help="weight decay")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--dropout", type=float, default=0.18074706609292976,
                        help="Dropout to use")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument("--eval", action='store_true',default=True,
                        help='If not set, we will only do the training part.')
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    print(args)
    
    set_seed(args)
    
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Remove duplicate edges
    # In PyG, this is a default pre-processing step for Reddit, see
    # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/reddit.py#L58
    g = data.graph
    g = dgl.add_self_loop(g)
    g = g.int().to(device)
    features, labels = features.to(device), labels.to(device)

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

    logger = Logger(args.runs, args)
    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))

            if not args.eval:
                continue

            train_acc, val_acc, test_acc = evaluate(model, features, labels, train_mask, val_mask, test_mask)
            logger.add_result(run, (train_acc, val_acc, test_acc))

            print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss.item(), train_acc, val_acc, test_acc))

        if args.eval:
            logger.print_statistics(run)

    if args.eval:
        logger.print_statistics()


if __name__ == '__main__':
    main()