
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle 
import os,sys,re

import random 
from random import shuffle
import numpy as np 

import itertools


# for each sequence, predict if it is valid promoter, and its expression level 

class Data (object): 
    def __init__ (self,filename,col_wanted,code_ix,make_extra):
        self.sequence = {} ## [ seq1, seq2 ... ]
        self.sequence_ix = {} ## [ [0,1,2,3] , ... ] indexing 0=A 1=T for example...
        self.sequence_feature = {} 
        self.label = {} 
        self.read_from_file(filename,col_wanted,make_extra)
        self.make_sequence_ix(code_ix)
        self.make_feature()
        print ('number obs. {}'.format(len(self.sequence)))

        self.sequence_ix_train = {} 
        self.sequence_ix_test = {}
        self.sequence_ix_dev = {} 

    def split_input_seq ( self,seqs ): # seqs = 'ACGTAGTT'
        seqs = list(itertools.chain.from_iterable(seqs))
        return seqs

    def make_sequence_ix (self,code_ix): # @code_x is dict indicating {A:0 T:1 ...}
        for key in self.sequence: 
            seqs = self.split_input_seq ( self.sequence [ key ] )
            seqs_ix = [ code_ix[i] for i in seqs ] ## convert to 0/1/2/3
            self.sequence_ix [ key ] =  seqs_ix 
    
    def read_from_file ( self,filename,col_wanted,make_extra ): 
        f = open ( filename, 'r', encoding='utf-8' )
        counter = 0
        for line in f: 
            line = line.split() ## want physical col 
            # if len(line[ col_wanted ]) > 500: 
            #     continue
            self.sequence[counter] = line [ col_wanted ] 
            self.label[counter] = int(line [ col_wanted + 1 ]) ## we need to edit the .txt 
            counter = counter + 1

            if (make_extra==1) & (self.label[counter-1]==0) : ## note "-1" for last counter point
                extra_seq = self.make_extra_negative_cases ( line [ col_wanted ] )
                self.sequence[counter] = extra_seq
                self.label[counter] = 0 
                counter = counter + 1

        f.close() 

    def make_feature(self): 
        for key in self.sequence: 
            self.sequence_feature[key] = self.make_feature_one_sequence ( self.sequence[key] )

    def make_feature_one_sequence (self,one_sequence): ## feature ordering A G C T
        ## we find the number of "left"A and "right"A ... same with the other nucleotides
        feature = torch.zeros ( 8, len(one_sequence) ) 
        for index,s in enumerate( one_sequence ): 
            for j in range((index-5),index): ## window span 5
                if j < 0:
                    continue
                if one_sequence[j]=='A':
                    feature[0,index] = feature[0,index] + 1
                if one_sequence[j]=='G':
                    feature[1,index] = feature[1,index] + 1
                if one_sequence[j]=='C':
                    feature[2,index] = feature[2,index] + 1
                if one_sequence[j]=='T':
                    feature[3,index] = feature[3,index] + 1
            ## update right
            for j in range((index+1), (index+6)): 
                if j >= len(one_sequence):
                    continue
                if one_sequence[j]=='A':
                    feature[4,index] = feature[4,index] + 1
                if one_sequence[j]=='G':
                    feature[5,index] = feature[5,index] + 1
                if one_sequence[j]=='C':
                    feature[6,index] = feature[6,index] + 1
                if one_sequence[j]=='T':
                    feature[7,index] = feature[7,index] + 1
        return feature / 5 ## scale down 

    def make_extra_negative_cases ( self, sequence ) :  
        ## for negative cases, swap the position of the snp 
        sequence = self.split_input_seq (sequence)  # @sequence must be an array.
        shuffle ( sequence )
        return "".join(i for i in sequence)
        

    def make_train_dev_test_random (self,num_train,num_dev): 
        person = list ( self.sequence_ix.keys() )
        shuffle(person) 
        for p in person[0:num_train]: 
            self.sequence_ix_train [p] = self.sequence_ix [p]
        for p in person[num_train:(num_train+num_dev)]: 
            self.sequence_ix_dev [p] = self.sequence_ix [p]
        for p in person[(num_train+num_dev): len(person) ]: 
            self.sequence_ix_test [p] = self.sequence_ix [p]
        ##

