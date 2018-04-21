from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

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
            nn.Linear( lstm_dim, self.num_classes) ) ## so we can apply nn.Linear to each entry in the batch 
    
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

        # print (lstm_out) # we have: #sentence x word_number x lstm_dim
        lstm_out, _ = torch.max( lstm_out, dim=1 ) ## because, filled by row... conv_emb_dim x batch_size 
        lstm_out = self.dense1 ( lstm_out )
        
        return lstm_out

