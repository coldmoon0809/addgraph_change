# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: train.py
# @time: 2020/03/13
import os
import time
import tqdm
import math
import itertools
import argparse
from framwork.snapshot import *
from framwork.model import *
from framwork.negative_sample import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--sample_rate', type=float, default=0.25, help='Sample sample_rate percent from initial edges.')
parser.add_argument('--ini_graph_percent', type=float, default=0.5, help='Train and test data percent.')
parser.add_argument('--anomaly_percent', type=float, default=0.05,
                    help='Anomaly injection with proportion of anomaly_percent.')
parser.add_argument('--snapshots_', type=int, default=1700, help='The snapshot size .')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50, help='Number of hidden units.')
parser.add_argument('--nmid1', type=int, default=70, help='Number of nmid1 units.')
parser.add_argument('--nmid2', type=int, default=100, help='Number of nmid2 units.')
parser.add_argument('--beta', type=float, default=3.0, help='Hyper-parameters in the score function.')
parser.add_argument('--mui', type=float, default=0.5, help='Hyper-parameters in the score function.')
parser.add_argument('--gama', type=float, default=0.6, help='Parameters in the score function.')
parser.add_argument('--w', type=int, default=3, help='Hyper-parameters in the score function.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()


# Load data
data_path = '/home/wasn/Addgraph/Digg_U_Addgraph_change/munmun_digg_reply/out.munmun_digg_reply'
Net1 = ConvGRU(in_channels=args.snapshots_,out_channels=args.hidden,kernel_size=3,stride=1,padding=0, dropout=args.dropout)
Net2 = HCA(hidden=args.hidden, dropout=args.dropout)
Net3 = GCN(nfeat=args.hidden, nmid1=args.nmid1, nmid2=args.nmid2, nhid=args.hidden, dropout=args.dropout)
Net4 = Score(beta=args.beta, mui=args.mui, hidden=args.hidden, dropout=args.dropout)
N_S = negative_sample()

optimizer = optim.Adam(itertools.chain(Net1.parameters(), Net2.parameters(), Net3.parameters(), Net4.parameters()),
                      lr=args.lr,
                      )  # weight_decay=args.weight_decay
# snapshots_train, l_train, snapshots_test, l_test, nodes, n_train = snapshot(data_path=data_path, sample_rate=args.sample_rate,
#                                                                    ini_graph_percent=args.ini_graph_percent,
#                                                                    anomaly_percent=args.anomaly_percent,
#                                                                    snapshots_=args.snapshots_)
# np.savez("snapshot_25_5a_17.npz",snapshots_train = snapshots_train, l_train = l_train, snapshots_test = snapshots_test, l_test = l_test, nodes = nodes, n_train = n_train)
snapshots=np.load("snapshot_25_5a_17.npz", allow_pickle=True)
snapshots_train, l_train, nodes, n_train=snapshots['snapshots_train'], snapshots['l_train'], snapshots['nodes'], snapshots['n_train']
l_train = int(l_train)
nodes = int(nodes)
n_train = int(n_train)


if args.cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('OK')
else:
    torch.set_default_tensor_type(torch.FloatTensor)


def train():
    t = time.time()
    Net1.train()
    Net2.train()
    Net3.train()
    Net4.train()
    N_S.train()

    # optimizer.zero_grad()
    for epoch in range(args.epochs):
        # snapshots_train = snapshots_train.cuda()
        H_list = torch.zeros(1, nodes, args.hidden)
        H_ = torch.zeros((args.w, nodes, args.hidden))
        for k in range(args.w - 1):
            H_list = torch.cat([H_list, torch.zeros(nodes, args.hidden).unsqueeze(0)], dim=0)
        stdv = 1. / math.sqrt(H_list[-1].size(1))
        H_list[-1][:n_train, :].data.uniform_(-stdv, stdv)
        adj = torch.zeros((nodes, nodes))
        loss_a = torch.zeros(1)
        for i in range(l_train):
            optimizer.zero_grad()
            # snapshot=snapshots_train[i]
            snapshot = torch.from_numpy(snapshots_train[i])
            H = H_list[-1]
            for j in range(args.w):
                H_[j] = H_list[-args.w + j]
            adj, Adj = update_adj(adj=adj, snapshot=snapshot, nodes=nodes)
            Adjn = normalize_adj(Adj + torch.eye(Adj.shape[0]))
            # adj_ = torch.from_numpy(adjn)
            if args.cuda:
                Net1.cuda()
                Net2.cuda()
                Net3.cuda()
                Net4.cuda()
                N_S.cuda()
                H = H.cuda()
                Adjn = Adjn.cuda()
                H_ = H_.cuda()
                snapshot = snapshot.cuda()
            H, H_, Adjn, snapshot = Variable(H), Variable(H_), Variable(Adjn), Variable(snapshot)
            current = Net3(x=H, adj=Adjn, Adj=Adj)
            short = Net2(C=H_)
            Hn = Net1(current=current, short=short)
            H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)
            n_loss = N_S(adj=adj, Adj=Adj, snapshot=snapshot, H=Hn, f=Net4, arg=args.cuda)
            loss1 = args.weight_decay * (Net1.loss() + Net2.loss() + Net3.loss() + Net4.loss())
            lens = n_loss.shape[0]
            zero = torch.zeros(1)
            loss2 = torch.zeros(1)
            for m in range(lens):
                count = n_loss[m]
                loss2 = loss2 + torch.where((args.gama + count) >= 0, (args.gama + count), zero)
            loss_a = loss_a + loss1 / (l_train) + loss2 / (l_train * lens)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            print(i)
            print('Loss of {}'.format(epoch), 'epoch,{}'.format(i), 'snapshot,loss:{}'.format(loss.item()))
            
        print('The average loss of {}'.format(epoch), 'epoch is :{}'.format(loss_a.item()))
        print(time.time() - t)
        writer1 = SummaryWriter('runs/R_Adam_w3_loss')
        # writer1 = SummaryWriter('runs/Adam_w3_loss')
        writer1.add_scalar('loss_avarage', loss_a.item(), epoch)
        print('===> Saving models...')
        state = {'Net1': Net1.state_dict(), 'Net2':
            Net2.state_dict(), 'Net3': Net3.state_dict(),
                 'Net4': Net4.state_dict(),
                'H_list': H_list, 'loss_a': loss_a, 'epoch': epoch}
        # print(n_loss)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        dir = './checkpoint/NEW_Sparse_S17_Adam_lr_0.001_w_3_epoch{}.pth'.format(epoch)
        torch.save(state, dir)


    adj = {'adj': adj}
    dir = './checkpoint/adj.pth'
    torch.save(adj, dir)
    print('Finish')
    

if __name__ == "__main__":
    train()
