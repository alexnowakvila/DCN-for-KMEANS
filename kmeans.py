
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

class Logger():
    dicc = {}
    def add(self, name, val):
        if name in self.dicc:
            lis = self.dicc[name]
            lis.append(val)
            self.dicc[name] = lis
        else:
            self.dicc[name] = [val]
    def empty(self, name):
        self.dicc[name] = []
    def empty_all(self):
        self.dicc = {}
    def get(self, name):
        return self.dicc[name]

def plot_train_logs(cost_train):
    plt.figure(1, figsize=(8,6))
    plt.clf()
    iters = range(len(cost_train))
    plt.plot(iters, cost_train, 'b')
    plt.xlabel('iterations')
    plt.ylabel('Average Mean cost')
    plt.title('Average Mean cost Training')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    path = os.path.join('plots/logs', 'training.png') 
    plt.savefig(path)


def plot_clusters(num, e, centers, points, fig):
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
        plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'o', c=rgb2hex(c))
        center = centers[i]
        plt.plot([center.data[0]], [center.data[1]], '*', c=rgb2hex(c))
    plt.title('clustering')
    plt.savefig('./plots/clustering_it_{}.png'.format(num))
    
    

def create_input(points, sigma2):
    bs, N, _ = points.size() #points has size bs,N,2
    OP = torch.zeros(bs,N,N,4).type(dtype)
    E = torch.eye(N).type(dtype).unsqueeze(0).expand(bs,N,N)
    OP[:,:,:,0] = E
    W = points.unsqueeze(1).expand(bs,N,N,2) - points.unsqueeze(2).expand(bs,N,N,2)
    dists2 = (W * W).sum(3)
    dists = torch.sqrt(dists2)
    W = torch.exp(-dists2 / sigma2)
    OP[:,:,:,1] = W
    D = E * W.sum(2,True).expand(bs,N,N)
    OP[:,:,:,2] = D
    U = (torch.ones(N,N).type(dtype)/N).unsqueeze(0).expand(bs,N,N)
    OP[:,:,:,3] = U
    OP = Variable(OP)
    x = Variable(points)
    Y = Variable(W.clone())
    return (OP, x, Y), dists

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

def update_input(input, dists, sample, sigma2):
    OP, x, Y = input
    bs = x.size(0)
    N = x.size(1)
    sample = sample.float()
    mask = sample.unsqueeze(1).expand(bs,N,N)*sample.unsqueeze(2).expand(bs,N,N)
    mask += (1-sample).unsqueeze(1).expand(bs,N,N)*(1-sample).unsqueeze(2).expand(bs,N,N)
    U = (OP.data[:,:,:,3]>0).float()*mask
    
    W = dists*U
    Wm = W.max(2,True)[0].expand_as(W).max(1,True)[0].expand_as(W)
    W = W / Wm.clamp(min=1e-6) * np.sqrt(2)
    W = torch.exp(- W*W / sigma2)
    
    OP[:,:,:,1] = Variable(W)
    D = OP.data[:,:,:,0] * OP.data[:,:,:,1].sum(2,True).expand(bs,N,N)
    OP[:,:,:,2] = Variable(D)
    
    U = U / U.sum(2,True).expand_as(U)
    OP[:,:,:,3] = Variable(U)
    Y = Variable(OP[:,:,:,1].data.clone())
    return OP, x, Y

def compute_variance(e, probs):
    bs, N = probs.size()
    variance = Variable(torch.zeros(bs).type(dtype))
    for i in range(e.max()+1):
        mask = Variable((e == i).float())
        Ns = mask.sum(1).clamp(min=1)
        masked_probs = probs*mask
        probs_mean = (masked_probs).sum(1) / Ns
        v = (masked_probs*masked_probs).sum(1) / Ns - probs_mean*probs_mean
        variance += v
    return variance

def compute_reward(e, K, points):
    bs, N, _ = points.size()
    reward2 = Variable(torch.zeros(bs).type(dtype))
    reward3 = Variable(torch.zeros(bs).type(dtype))
    c = []
    for k in range(2**K):
        mask = Variable((e == k).float()).unsqueeze(2).expand_as(points)
        N1 = mask.sum(1)
        center = points*mask
        center = center.sum(1) / N1.clamp(min=1)
        c.append(center[0])
        subs = ((points-center.unsqueeze(1).expand_as(points)) * mask)
        subs2 = (subs * subs).sum(2).sum(1) / N
        subs3 = torch.abs(subs * subs * subs).sum(2).sum(1) / N
        reward2 += subs2
        reward3 += subs3
    return reward2, reward3, c

def execute(points, K, n_samples, sigma2, reg_factor, mode='test'):
    bs, N, _ = points.size()
    e = torch.zeros(bs, N).type(dtype_l)
    input, dists = create_input(points.data, sigma2)
    loss_total = Variable(torch.zeros(1).type(dtype))
    for k in range(K):
        scores,_ = gnn(input)
        probs = F.sigmoid(scores)
        if mode == 'train':
            variance = compute_variance(e, probs)
            variance = variance.sum() / bs
            Lgp = Variable(torch.zeros(n_samples, bs).type(dtype))
            Reward2 = Variable(torch.zeros(n_samples, bs).type(dtype))
            Reward3 = Variable(torch.zeros(n_samples, bs).type(dtype))
            for i in range(n_samples):
                Samplei, Lgp[i] = sample_one(probs, 'train')
                Ei = e*2 + Samplei.long()
                Reward2[i], _,_ = compute_reward(Ei, k+1, points)
            baseline = Reward2.mean(0,True).expand_as(Reward3)
            loss = ((Reward2-baseline) * Lgp).sum(1).sum(0) / n_samples / bs
            loss_total = loss_total + loss - reg_factor*variance
            show_loss = Reward2.data.mean()
        sample, lgp = sample_one(probs, 'test')
        e = e*2 + sample.long()
        reward,_,c = compute_reward(e, k+1, points)
        if mode == 'test':
            show_loss = reward.data.mean()
        if k < K-1:
            input = update_input(input, dists, sample, sigma2)
    if mode == 'test':
        return e, show_loss, c
    else:
        return e, loss_total, show_loss, c


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
    N = 80
    clusters = 8
    clip_grad_norm = 40.0
    batch_size = 256
    num_features = 32
    num_layers = 20
    sigma2 = 1
    reg_factor = 0.00
    K = 3
    k_step = 0000
    n_samples = 10
    
    gen = Generator('/data/folque/dataset/', num_examples_train, num_examples_test, N, clusters)
    gen.load_dataset()
    num_iterations = 100000
    
    gnn = Split_GNN(num_features, num_layers, 5, dim_input=2)
    if args.load_file != '':
        gnn = load_model(args.load_file, gnn)
    optimizer = optim.RMSprop(gnn.parameters(), lr=1e-3)
    
    test = args.test
    if test:
        num_iterations = num_examples_test // batch_size
    
    
    log = Logger()
    start = time.time()
    for it in range(num_iterations):
        batch = gen.sample_batch(batch_size, is_training=not test)
        points, target = batch
        if k_step > 0:
            k = min(K,1+it//k_step)
        else:
            k = K
        
        e, loss, show_loss, c = execute(points, k, n_samples, sigma2, reg_factor, mode='train')
        log.add('show_loss', show_loss)
        
        if not test:
            gnn.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(gnn.parameters(), clip_grad_norm)
            optimizer.step()
            
        if not test:
            if it%50 == 0:
                elapsed = time.time()-start
                print('iteration {}, var {}, loss {}, elapsed {}'.format(it, show_loss, loss.data.mean(), elapsed))
                plot_clusters(it, e, c, points.data, 0)
                #out1 = ['---', it, loss, w, wt, tw, elapsed]
                #print(template_train1.format(*info_train))
                #print(template_train2.format(*out1))
                
                start = time.time()
            if it%300 == 0 and it > 0:
                plot_train_logs(log.get('show_loss'))
                if args.save_file != '':
                    save_model(args.save_file, gnn)
        
    if test:
        a = 1
        #ensenyar resultats
            
    



















