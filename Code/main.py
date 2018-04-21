
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

from data import * 
from model import *
from train import *
from helper import * 

random.seed(89101112) # seed 
torch.manual_seed(999) # seed
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)
    

code_ix = { 'PADDING':0,'A':1,'T':2,'C':3,'G':4 }
num_of_word=len(code_ix)

data = pickle.load ( gzip.open("E:/eColiPromoterData/Peak_Prediction/dynamic_max_b/data_make_extra.pickle.gzip", "rb") )

num_epoch = 30
save_path = 'E:/eColiPromoterData/Peak_Prediction/dynamic_max_b/'
if os.path.exists(save_path) == False: 
    os.mkdir(save_path)

## 
is_feature = True ## use extra left/right nucleotide counting (or any other feature)
feature_dim= 8

batch_size = 16
num_of_classes=2

dropout_rate = .2

word_emb_dim=50 # equal contribution as feature_dim ? 
lstm_hidden_dim=50 ## should be word_emb_dim+feature_dim ? 

conv_row=lstm_hidden_dim ## convolution layer, not used yet 
conv_col=10

batch_size = 32


number_segment = 10 ## for segmenting the max-pool step 


# assign people to batch 
total_number_sample = len(data.sequence)
num_train = int (total_number_sample*.95)
num_dev = total_number_sample - num_train
data.make_train_dev_test_random ( num_train, num_dev ) 

# now we actually make the batches 
batch_train, batch_train_label = make_format_batch (batch_size,data.sequence_ix_train,data.label,data.sequence_feature)
batch_dev, batch_dev_label = make_format_batch (batch_size,data.sequence_ix_dev,data.label,data.sequence_feature)

# print ('\nsample of batch\n')
# print (batch_train_label[0])
# print (batch_dev_label[0])

# make model 
model = biLSTM_maxpool ( num_of_classes,dropout_rate,True,num_of_word,word_emb_dim,lstm_hidden_dim,batch_size,feature_dim,is_feature)

# train model 
train_model = TrainModel (model,num_of_classes,num_epoch,save_path,batch_size,True)
train_model.trainNepoch(batch_train,batch_train_label,batch_dev,batch_dev_label)
