#!/usr/bin/env python
from __future__ import print_function
import cPickle as pickle
import six
import argparse
import time
import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Variable, FunctionSet
from chainer import serializers

import net

import mean_squared_error

print ("Loading...")
data = six.moves.cPickle.load(open("data/training.pickles","rb"))
allKeys=data.keys()
allKeys.remove("Image")


"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""

parser = argparse.ArgumentParser(description='Facial recognition with Chainer')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--epoch', '-e', default=200, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=2000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
n_units = args.unit

print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# Prepare dataset
N = 3500



x_train, x_test = np.split(data['Image'].astype(np.float32),   [N])
#y_train, y_test = np.split(data[tgt].astype(np.float32).reshape((data[tgt].size,1))/96., [N])
y_all = np.zeros(len(allKeys) * len(data["Image"]) )
for i,k in enumerate(allKeys):
    #print(k)
    data[k] = data[k] / 96.
    data[k][np.isnan(data[k])] = np.average(data[k][~np.isnan(data[k])])
    y_all[i*len(data[k]):(i+1)*len(data[k])] = data[k]
y_all=y_all.reshape((len(allKeys),len(y_all)/len(allKeys)))
y_all = y_all.T
y_all = y_all.astype(np.float32)

y_train, y_test = np.split(y_all, [N])

N_test = y_test.size

xp = np

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

model = FunctionSet(l1=F.Linear(96*96, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, n_units),
                    l4=F.Linear(n_units, n_units),
                    l5=F.Linear(n_units, n_units/2),
                    l6=F.Linear(n_units/2, len(allKeys)))

def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x )), train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    h3 = F.dropout(F.relu(model.l3(h2)), train=train)
    h4 = F.dropout(F.relu(model.l4(h3)), train=train)
    h5 = F.dropout(F.relu(model.l5(h4)), train=train)
    y  = model.l6(h5)
    return F.mean_squared_error(y, t), y

# Setup optimizer
optimizer = optimizers.AdaDelta(rho=0.90)
optimizer.setup(model)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)
    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize):
        x = xp.asarray(x_train[perm[i:i + batchsize]])
        t = xp.asarray(y_train[perm[i:i + batchsize]])
        optimizer.zero_grads()
        loss, prod = forward(x,t)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        #print (i,sum_loss)

    print ('train mean loss={}'.format(sum_loss / N))
    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)

