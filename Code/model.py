from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import math 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class biLSTM_maxpool(nn.Module): 

    def __init__(self, num_classes, dropout_rate, doCuda, num_of_word, word_emb_dim, lstm_dim, batch_size, feature_dim=0, is_feature=False ): # @num_of_word is vocab size. 
        
        super().__init__()

        self.doCuda = doCuda
        self.batch_size = batch_size 
        self.is_feature = is_feature

        self.word_emb_dim = word_emb_dim # @word_emb_dim is emb size (i.e. 300 in GloVe)
        self.embedding = nn.Embedding(num_of_word, word_emb_dim) ## @embedding, row=word, col=some dimension value 
        
        self.lstm_dim = lstm_dim ## lstm layer 
        if is_feature == False:
            feature_dim = 0 

        lstm_input_dim = word_emb_dim + feature_dim ## note that if we have feature, then the input to lstm must include these features 
        self.lstm = nn.LSTM( lstm_input_dim , lstm_dim // 2, bidirectional=True, num_layers=1, batch_first=True ) ## divide 2 because each direction needs "half"
        
        self.hidden_state = self.init_hidden()

        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # self.dense1 = nn.Sequential (
        #     nn.Dropout(dropout_rate), 
        #     nn.Linear(lstm_dim, lstm_dim//2),
        #     nn.Tanh(),
        #     nn.Linear( lstm_dim//2, self.num_classes) ) ## so we can apply nn.Linear to each entry in the batch 

        self.dense1 = nn.Sequential (
            nn.Dropout(dropout_rate), 
            nn.Linear( self.lstm_dim, self.num_classes) ) ## so we can apply nn.Linear to each entry in the batch 
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.doCuda == True:
            return (	Variable(torch.randn(2, self.batch_size, self.lstm_dim // 2)).cuda() ,
                        Variable(torch.randn(2, self.batch_size, self.lstm_dim // 2)).cuda() ) # NOTICE. "2" for num_layers when using bi-directional (with 1 layer indication)
        else:
            return (	Variable(torch.randn(2, self.batch_size, self.lstm_dim // 2)) ,
                        Variable(torch.randn(2, self.batch_size, self.lstm_dim // 2)) )

    def init_embedding_weights(self, pretrained_weight): # @pretrained_weight has to be a numpy type 
        print ('init pre-trained vectors to embedding layer. we do not freeze embedding during training. \n')
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight)) ## Initialize weight parameters for the embedding layer. 

    def forward(self, sentence,seq_lengths,feature=None): # @self.embedding(sentence) is 2d tensor. row = a word, col = each dim (i.e. 300 cols). 
        embeds = self.embedding(sentence.cuda()) # @sentence is an array [word-ix1, word-ix2 ...]
        if self.is_feature:
            # print (feature.transpose(1,2))
            # print (embeds)
            feat = Variable(feature.transpose(1,2)).cuda() 
            embeds = torch.cat( (embeds,feat) ,dim=2 )

        embeds = pack_padded_sequence(embeds, seq_lengths.numpy() , batch_first=True )
        lstm_out, self.hidden_state = self.lstm(embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out,batch_first=True ) # unpack your output if required

        # print (lstm_out) # we have: #batch x sequence_len x lstm_dim
        lstm_out, _ = torch.max( lstm_out, dim=1 ) ## because, filled by row... conv_emb_dim x batch_size 
        lstm_out = self.dense1 ( lstm_out )
        
        return lstm_out



class biLSTM_dynamic_maxpool(biLSTM_maxpool): 

    def __init__(self, num_classes, dropout_rate, doCuda, num_of_word, word_emb_dim, lstm_dim, num_segment, batch_size, feature_dim=0, is_feature=False ): # @num_of_word is vocab size. 
        super().__init__(num_classes, dropout_rate, doCuda, num_of_word, word_emb_dim, lstm_dim, batch_size, feature_dim=feature_dim, is_feature=is_feature)
        
        self.num_segment = num_segment 
        self.max_layer_dim = self.lstm_dim*self.num_segment # this is #of_lstm_dim, if we do simple max-pool 

        # self.dense1 = nn.Sequential (
        #     nn.Dropout(dropout_rate), 
        #     nn.Linear(lstm_dim, lstm_dim//2),
        #     nn.Tanh(),
        #     nn.Linear( lstm_dim//2, self.num_classes) ) ## so we can apply nn.Linear to each entry in the batch 

        self.dense1 = nn.Sequential (
            nn.Dropout(dropout_rate), 
            nn.Linear( self.max_layer_dim, self.num_classes) ) ## so we can apply nn.Linear to each entry in the batch 

    def max_every_k_seg( self,X) :  ## X is matrix  #batch x lstm_dim x sequence_len 
        window = math.ceil(X.size(2)*1.0/self.num_segment) ## X.size(2) number words in X
        padding_needed = window*self.num_segment - X.size(2)
        if padding_needed > 0: 
            zero_pad_dim = ( 1, X.size(1), padding_needed )
            temp = Variable(torch.zeros(zero_pad_dim) ).cuda()
            X = torch.cat ( ( X, temp) , dim=2 )
        #
        maxpool_op = nn.MaxPool1d (window, stride=window, padding=0)
        max_x = maxpool_op ( X )
        return max_x.view( 1, max_x.numel() ) 

    def take_max ( self, lstm_out, seq_lengths ): ## for each sample in the batch, we take the max 
        dynamic_max_layer = Variable( torch.zeros( lstm_out.size(0),1,self.max_layer_dim ) ).cuda() 
        for b in range(len(lstm_out)): 
            X = lstm_out[b,0:seq_lengths[b],:] # do not need the dummy encoding (aka the padding)
            X = X.transpose(0,1) # lstm_dim x sequence_len 
            X = X.unsqueeze(0)
            dynamic_max_layer[b] = self.max_every_k_seg(X) 
        return dynamic_max_layer

    def forward(self, sentence,seq_lengths,feature=None): # @self.embedding(sentence) is 2d tensor. row = a word, col = each dim (i.e. 300 cols). 
        embeds = self.embedding(sentence.cuda()) # @sentence is an array [word-ix1, word-ix2 ...]
        if self.is_feature:
            # print (feature.transpose(1,2))
            # print (embeds)
            feat = Variable(feature.transpose(1,2)).cuda() 
            embeds = torch.cat( (embeds,feat) ,dim=2 )

        embeds = pack_padded_sequence(embeds, seq_lengths.numpy() , batch_first=True )
        lstm_out, self.hidden_state = self.lstm(embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out,batch_first=True ) # unpack your output if required

        # print (lstm_out) # we have: #batch x sequence_len x lstm_dim
        lstm_out = self.take_max( lstm_out,seq_lengths ) ## because, filled by row... conv_emb_dim x batch_size 
        lstm_out = self.dense1 ( lstm_out ) ## 3D, #batch x #classes 
        lstm_out = lstm_out.view(lstm_out.size(0),self.num_classes) ## make into 2D
        return lstm_out

