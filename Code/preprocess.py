## 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle , gzip
import os,sys,re

import random 
from random import shuffle
import numpy as np 

import itertools

from helper import * 
from data import * 

code_ix = { 'PADDING':0,'A':1,'T':2,'C':3,'G':4 }
# code_ix = { 'A':0,'T':1,'C':2,'G':3 }

data = Data( '/u/flashscratch/d/datduong/eColiPromoterData/Peak_Prediction/BBD_peakCalls_both_format_shuf.txt', 0, code_ix, 1 )
pickle.dump (data, gzip.open("/u/flashscratch/d/datduong/eColiPromoterData/Peak_Prediction/data_make_extra.pickle.gzip", "wb") )

print ( data.sequence[0] ) 
print ( data.sequence_ix[0] )
print ( data.sequence_feature[0] ) 
print ( data.sequence_feature[0][:,0] ) 
print ( data.label[0] )

# # assign people to batch 
# total_number_sample = len(data.sequence)
# num_train = int (total_number_sample*.95)
# num_dev = total_number_sample - num_train
# data.make_train_dev_test_random ( num_train, num_dev ) 

# # now we actually make the batches 
# batch_size = 16
# batch_train, batch_train_label = make_format_batch (batch_size,data.sequence_ix_train,data.label,data.sequence_feature)
# batch_dev, batch_dev_label = make_format_batch (batch_size,data.sequence_ix_dev,data.label,data.sequence_feature)

# print ('\nsample of 1 batch\n')
# print (batch_train[0])

# pickle.dump (batch_train, gzip.open("/u/flashscratch/d/datduong/eColiPromoterData/Peak_Prediction/batch_train.pickle.gzip", "wb") )
# pickle.dump (batch_train_label, gzip.open("/u/flashscratch/d/datduong/eColiPromoterData/Peak_Prediction/batch_train_label.pickle.gzip", "wb") )

# pickle.dump (batch_dev, gzip.open("/u/flashscratch/d/datduong/eColiPromoterData/Peak_Prediction/batch_dev.pickle.gzip", "wb") )
# pickle.dump (batch_dev_label, gzip.open("/u/flashscratch/d/datduong/eColiPromoterData/Peak_Prediction/batch_dev_label.pickle.gzip", "wb") )

