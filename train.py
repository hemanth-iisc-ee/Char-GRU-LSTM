from __future__ import division

import os
import sys  
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import pickle
import pdb

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output,(hidden_state[0].detach(), hidden_state[1].detach())

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUNet, self).__init__()
        self.embedding = nn.Embedding(input_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h):
        e = self.embedding(x)
        out, h = self.gru(e, h)
        out = self.fc(out)
        return out, h.detach()
    
    # def init_hidden(self, batch_size=1):
    #     weight = next(self.parameters()).data
    #     hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
    #     return hidden



parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_file', type=str, default='data/sherlock.txt', help='path to input text file')
parser.add_argument('-s','--sequence_len', type=int, default=200, help='sequence length')
parser.add_argument('-b','--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('-l','--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

use_cuda = opt.use_cuda
cuda = torch.cuda.is_available() and opt.use_cuda
device = torch.device('cuda') if cuda else torch.device('cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


data = open(opt.input_file, 'r').read()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)


print("----------------------------------------")
print("Data has {} characters, {} unique".format(data_size, vocab_size))
print("----------------------------------------")   

# char to index and index to char maps
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# convert data from chars to indices
data = list(data)
for i, ch in enumerate(data):
    data[i] = char_to_ix[ch]

# data tensor on device
data = torch.tensor(data).to(device)
data = torch.unsqueeze(data, dim=1)

# model = GRUNet(vocab_size, 512, vocab_size, 1).to(device)
model = RNN(vocab_size,vocab_size,512,1).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

alpha = 0.1
step = 100
EPOCHS = 1
loss_prev_val = []
loss_max = -np.inf
prev_epoch = 1
for epoch in range(1,EPOCHS+1):
    # random starting point (1st 100 chars) from data to begin
    data_ptr = np.random.randint(100)
    n = 0
    running_loss = 0
    hidden_state = None
    while True:
    	# random starting point (1st 100 chars) from data to begin
        input_seq = data[data_ptr : data_ptr+opt.sequence_len]
        target_seq = data[data_ptr+1 : data_ptr+opt.sequence_len+1]
        pdb.set_trace()
        # forward pass
        output, hidden_state = model(input_seq, hidden_state)
        # compute loss
        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
        
        # compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        # if at end of data : break
        if data_ptr + 2*opt.sequence_len + 1 > data_size:
            break
        if n%step == 0:
        	# print(n,'/', data_size, ', loss:', loss.item())
        	
        	plt.figure(1)
        	if not loss_prev_val:
        		n1, l1, l1_smooth = 0,loss.item(),loss.item()
        		plt.plot(n1, l1, color='b')
        	else:
        		n0,n1 = loss_prev_val[0], loss_prev_val[0]+(step*opt.sequence_len)
        		l0,l1 = loss_prev_val[1], loss.item()
        		plt.plot([n0,n1], [l0,l1],color='b')
        		l0_smooth, l1_smooth = loss_prev_val[2],(1-alpha)*loss_prev_val[2] + alpha*l1
        		plt.plot([n0,n1], [l0_smooth,l1_smooth],color='r')
        	if l1 >= loss_max:
        		loss_max = l1
        	loss_prev_val = [n1,l1,l1_smooth]
        	if prev_epoch != epoch:
        		plt.plot([n1,n1],[0,loss_max],color='k',linestyle='-.')
        		prev_epoch = epoch

        	# print(n1,'/', data_size, ', loss:', loss.item())
        	plt.grid(True)
        	plt.show(block=False)
        	plt.pause(0.01)
        # update the data pointer
        data_ptr += opt.sequence_len
        n +=1

    # pdb.set_trace()

# random character from data to begin
data_ptr = 0
hidden_state = None
rand_index = np.random.randint(data_size-1)
input_seq = data[rand_index : rand_index+1]
op_seq_len = 1000

print("----------------------------------------")
while True:
    # forward pass
    output, hidden_state = model(input_seq, hidden_state)
    
    # construct categorical distribution and sample a character
    output = F.softmax(torch.squeeze(output), dim=0)
    dist = Categorical(output)
    index = dist.sample()
    
    # print the sampled character
    print(ix_to_char[index.item()], end='')
    
    # next input is current output
    input_seq[0][0] = index.item()
    data_ptr += 1
    
    if data_ptr > op_seq_len:
        break
    
print("\n----------------------------------------")