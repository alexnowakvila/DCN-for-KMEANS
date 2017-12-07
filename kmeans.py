
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os
import sys
import time
import math
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex

from model import Split_GNN
from data_generator import Generator



parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--save_file', nargs='?', const=1, type=str, default='')
parser.add_argument('--load_file', nargs='?', const=1, type=str, default='')
parser.add_argument('--output_file', nargs='?', const=1, type=str, default='')
args = parser.parse_args()


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

template_train1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} '
template_train2 = '{:<10} {:<10} {:<10.3f} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.3f} '
template_train3 = '{:<10} {:<10} {:<10} {:<10.5f} {:<10.5f} {:<10.5f} {:<10} \n'
info_train = ['TRAIN', 'iteration', 'loss', 'samples', 'best_smpl', 'trivial', 'elapsed']


if args.output_file != '':
    class Logger2(object):
        def __init__(self, path):
            self.terminal = sys.stdout
            self.log = open(path, 'a')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass    

    sys.stdout = Logger2(args.output_file)


def create_input(points, sigma2):
    bs, N, _ = points.size() #points has size bs,N,2
    OP = torch.zeros(bs,N,N,4).type(dtype)
    E = torch.eye(N).type(dtype).unsqueeze(0).expand(bs,N,N)
    OP[:,:,:,0] = E
    W = points.unsqueeze(1).expand(bs,N,N,2) - points.unsqueeze(2).expand(bs,N,N,2)
    W = torch.exp(-(W * W).sum(3) / sigma2)
    OP[:,:,:,1] = W
    D = E * W.sum(2,True).expand(bs,N,N)
    OP[:,:,:,2] = D
    U = (torch.ones(N,N).type(dtype)/N).unsqueeze(0).expand(bs,N,N)
    OP[:,:,:,3] = U
    OP = Variable(OP)
    x = Variable(points)
    Y = Variable(W.clone())
    return OP, x, Y

def sample_one(probs, mode='test'):
    probs = 1e-6 + probs*(1 - 2e-6) # to avoid log(0)
    if mode == 'train':
        rand = torch.zeros(*probs.size()).type(dtype)
        nn.init.uniform(rand)
    else:
        rand = torch.ones(*probs.size()).type(dtype) / 2
    bin_sample = probs > Variable(rand)
    sample = bin_sample.clone().type(dtype)
    log_probs_samples = (sample*torch.log(probs) + (1-sample)*torch.log(1-probs)).sum(1)
    return bin_sample.data, log_probs_samples

def update_input(input, sample):
    OP, x, Y = input
    bs = x.size(0)
    N = x.size(1)
    sample = sample.float()
    mask = sample.unsqueeze(1).expand(bs,N,N)*sample.unsqueeze(2).expand(bs,N,N)
    mask += (1-sample).unsqueeze(1).expand(bs,N,N)*(1-sample).unsqueeze(2).expand(bs,N,N)
    OP[:,:,:,1] = Variable(mask*OP.data[:,:,:,1])
    D = OP.data[:,:,:,0] * OP.data[:,:,:,1].sum(2,True).expand(bs,N,N)
    OP[:,:,:,2] = Variable(D)
    U = (OP.data[:,:,:,3]>0).float()*mask
    U = U / U.sum(2,True).expand_as(U)
    OP[:,:,:,3] = Variable(U)
    Y = OP[:,:,:,1].clone()
    return OP, x, Y
    

def compute_loss(e, K, lgp, points):
    bs = points.size(0)
    loss = Variable(torch.zeros(1).type(dtype))
    variances = Variable(torch.zeros(bs).type(dtype))
    for k in range(K):
        mask = Variable((e == k).float()).unsqueeze(2).expand_as(points)
        N1 = mask.sum(1)
        center = points*mask
        center = center.sum(1) / N1.clamp(min=1)
        subs = ((points-center.unsqueeze(1).expand_as(points)) * mask)
        subs = (subs * subs).sum(2).sum(1)
        #baseline = subs.mean(0,True).expand_as(subs)
        loss = loss + (subs*lgp).sum(0) / bs
        variances += subs.sum(0) / bs
    return loss, variances

def execute(points, sigma2, K, mode='test'):
    bs, N, _ = points.size()
    e = torch.zeros(bs, N).type(dtype_l)
    input = create_input(points.data, sigma2)
    loss_total = Variable(torch.zeros(1).type(dtype))
    for k in range(K):
        scores,_ = gnn(input)
        sample, lgp = sample_one(F.sigmoid(scores), mode)
        e = e*2 + sample.long()
        loss, vs = compute_loss(e, k+1, lgp, points)
        loss_total = loss_total + loss
        if k < K-1:
            input = update_input(input, sample)
    return e, loss_total, vs

def plot_clusters(num, e, points, fig):
    plt.figure(0)
    plt.clf()
    plt.gca().set_xlim([-0.05,1.05])
    plt.gca().set_ylim([-0.05,1.05])
    clusters = e[fig].max()+1
    colors = cm.rainbow(np.linspace(0,1,clusters))
    for i in range(clusters):
        c = colors[i][:-1]
        mask = e[fig] == i
        x = torch.masked_select(points[fig,:,0], mask)
        y = torch.masked_select(points[fig,:,1], mask)
        plt.scatter(x.cpu().numpy(), y.cpu().numpy(), c=rgb2hex(c))
    plt.title('clustering')
    plt.savefig('./plots/clustering_it_{}.png'.format(num))
    

def save_model(path, model):
    torch.save(model.state_dict(), path)
    print('Model Saved.')

def load_model(path, model):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print('GNN successfully loaded from {}'.format(path))
        return model
    else:
        raise ValueError('Parameter path {} does not exist.'.format(path))

if __name__ == '__main__':
    
    num_examples_train = 20000
    num_examples_test = 1000
    N = 20
    clusters = 4
    clip_grad_norm = 40.0
    batch_size = 256
    num_features = 32
    num_layers = 10
    sigma2 = 0.5
    K = 1
    
    gen = Generator('/data/folque/dataset/', num_examples_train, num_examples_test, N, clusters)
    gen.load_dataset()
    num_iterations = 100000
    
    gnn = Split_GNN(num_features, num_layers, 5, dim_input=2)
    #if args.load_file != '':
    #    Knap = load_model(args.load_file, Knap)
    optimizer = optim.Adamax(gnn.parameters(), lr=1e-3)
    
    test = args.test
    if test:
        num_iterations = num_examples_test // batch_size
    
    start = time.time()
    for it in range(num_iterations):
        batch = gen.sample_batch(batch_size, is_training=not test)
        points, target = batch
        
        e, loss, vs = execute(points, sigma2, K, mode='train')
        
        if not test:
            gnn.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(gnn.parameters(), clip_grad_norm)
            optimizer.step()
        
        if not test:
            if it%50 == 0:
                print('iteration {}, var {}'.format(it,vs.data.mean()))
                
                e, loss, vs = execute(points, sigma2, K, mode='test')
                plot_clusters(it, e, points.data, 0)
                #out1 = ['---', it, loss, w, wt, tw, elapsed]
                #print(template_train1.format(*info_train))
                #print(template_train2.format(*out1))
                
                start = time.time()
            if it%1000 == 0 and it >= 0:
                if args.save_file != '':
                    save_model(args.save_file, Knap)
    if test:
        a = 1
        #ensenyar resultats
            




















