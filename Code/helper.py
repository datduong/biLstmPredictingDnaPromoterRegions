
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import time
import pickle 
import os,sys,re
import numpy as np 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

## 

def pad_sentences ( vectorized_seqs ): ## add in 0 padding at the end of the "batch of sentences"
    # get the length of each seq in your batch
    seq_lengths = torch.LongTensor(list (map(len, vectorized_seqs)) )
    # dump padding everywhere, and place seqs on the left.
    # NOTE: you only need a tensor as big as your longest sequence
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()  #.cuda()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    return seq_lengths,seq_tensor,perm_idx # 2d tensor, row=sentence col=word_indexing

def make_batch ( batch_size , data_dict ) : ## we log the id of the person 
    # @data_dict is something like @person_note_ix 
    batch = {} 
    batch_ix = -1
    for i,key in enumerate(data_dict) : 
        if i % batch_size == 0: ## add 1st observation to new batch 
            batch_ix = batch_ix + 1 ## starting a new batch, so add 1 to increase indexing
            batch[batch_ix] = [ key ]
        else: 
            batch[batch_ix].append( key )
    ##
    print ('make batch done... total batch number is {}'.format(batch_ix))
    return batch 

def format_batch_sep_lab (batch,data_sequence,data_label,data_feature) : ## default 0/1 classification 
    ## return a separted variable to contain label, because of hoffman2 not saving the file 
    batch_label = {} 
    for b in batch:  

        # using 'enumerate' twice will not give same ordering, best to make explicit indexing 
        explicit_index = { i:p for i,p in enumerate (batch[b]) } # 1:person 2:person 3:person 
        
        vectorized_seqs = [] 
        label_b = torch.FloatTensor( len(batch[b]) , 1 ) ## wrap Variable around this later.  
        for i in explicit_index: ## for each person, get the word-indexing 
            vectorized_seqs.append( data_sequence[ explicit_index[i] ] )
            label_b [i] = data_label[ explicit_index[i] ] # position "i" is for person explicit_index[i]

        seq_lengths,seq_tensor,perm_idx = pad_sentences ( vectorized_seqs ) ## @perm_idx must be used, the sentences must be order from longest to shortest 
        label_b = label_b[perm_idx] ## need to "reorder" the gold label
        
        ## not so elegant here ...
        person_order = {}
        for i,value in enumerate(perm_idx.numpy().tolist()): ## @i is the indexing physical location, @value is permutation ordering 
            person_order [ batch[b][value] ] = i   ## the people ordering. this is needed for back tracking 

        ## need to format the left/right feature.
        feature_b = torch.FloatTensor( len(batch[b]) , 8, seq_tensor.size(1) ) ## seq_tensor.size(1) is maximum len 
        for i in explicit_index : 
            feat_i = data_feature[ explicit_index[i] ] ## for the person in position i. 
            num_to_pad = seq_tensor.size(1) - feat_i.size(1) 
            if num_to_pad > 0: 
                feat_i = torch.cat ( (feat_i,torch.zeros( 8,num_to_pad )), dim=1 ) ## if num-pad is 0, then we have extra dim. 
            #
            feature_b[i] = feat_i 

        feature_b = feature_b[perm_idx] ## need to "reorder" 

        ## 
        batch [b] = {'seq_lengths':seq_lengths,'seq_tensor':seq_tensor,'seq_feature':feature_b,'person_order':person_order}
        batch_label [b] = {'gold_label':label_b,'person_order':person_order}

    return batch , batch_label


def make_format_batch (batch_size,data_sequence,data_label,data_feature): 
    batch = make_batch(batch_size,data_sequence)
    batch, batch_label = format_batch_sep_lab ( batch, data_sequence,data_label,data_feature) ## batch will be {1: {seq_lengths:[vec], 'seq_tensor':[vec] } }
    return batch, batch_label

