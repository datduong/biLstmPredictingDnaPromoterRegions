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

from model import *


class TrainModel (object):
    def __init__(self,model,num_diagnosis,num_epoch,save_path,batch_size,doCuda=True,lr=.1):

        self.batch_size = batch_size
        self.model = model
        self.num_epoch = num_epoch

        self.num_diagnosis = num_diagnosis
        self.loss_function = nn.CrossEntropyLoss( weight=torch.FloatTensor([0.75,1.5]) )

        self.lr = lr # learning rate. 
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-3)
        
        self.best_dev_loss = np.inf
        self.save_path = save_path

        self.doCuda = doCuda
        if self.doCuda==True: 
            print ('will use cuda') 
            self.model.cuda() 
            self.loss_function = self.loss_function.cuda() 

    def train1epoch (self,batch,batch_label,is_train): 

        total_loss = torch.Tensor([0])
        accuracy = 0.0 
        for batch_index in batch :

            ## step 1 
            if is_train == True:
                self.model.zero_grad() # Recall that torch *accumulates* gradients. Before passing in a new instance, you need to zero out the gradients from the old instance
                self.model.hidden_state = self.model.init_hidden() # detaching it from its history on the last instance.
            
            ## step 2 and 3
            prob = self.model.forward ( batch[batch_index]['seq_tensor'],batch[batch_index]['seq_lengths'],batch[batch_index]['seq_feature'] )  
            
            ## step 4
            label = Variable( torch.LongTensor( batch_label[batch_index]['gold_label'].numpy() ) ).cuda() # 2D tensor
            label = label.view(label.size(0)) ## 1D 
            loss = self.loss_function( prob , label )  # Compute your loss function. #.cuda()
            
            if is_train == True: 
                loss.backward() # Do the backward pass and update the gradient # retain_graph=True
                self.optimizer.step()

            total_loss += loss.cpu().data

            accuracy = accuracy + self.get_accuracy ( prob, label )
            # if batch_index == 0:
            #     print ('sample') 
            #     print (F.softmax(prob[0:5],dim=1))

        #
        # print ('total sum accuracy {}'.format(accuracy))
        return total_loss.numpy()[0],accuracy/len(batch)

    def trainNepoch ( self, batch_train,batch_train_label,batch_dev,batch_dev_label):
        
        print ('number epoch {}'.format(self.num_epoch))
        print ('number of batch in train set {}'.format(len(batch_train)))
        print ('number of batch in dev set {}'.format(len(batch_dev)))
        
        reportEvery = self.num_epoch // 5
        if (reportEvery == 0): ## avoid division by 0 when showing each step. 
            reportEvery = 1

        trackingLossTrain = np.array ( [np.inf] ) 
        trackingLossDev = np.array ( [np.inf] )
        for epoch in range(1,self.num_epoch+1):

            self.model.train()
            # print ("\n")

            start_time = time.time()
            total_loss,accuracy = self.train1epoch(batch_train,batch_train_label,True)
            elapsed_time = time.time() - start_time
            if ( (epoch % reportEvery) == 0 ) | ( epoch==1) : 
                print ('\nTRAINING-SET epoch {} , total loss {} , accuracy {} , learn_rate {} , time second {} '.format(epoch,total_loss,accuracy,self.lr,elapsed_time) )
            trackingLossTrain = np.append( trackingLossTrain, total_loss)
        
            ##
            ## apply the model on the development-set 

            self.model.eval() 

            start_time = time.time()
            lossOnDevSet,accuracy = self.train1epoch(batch_dev,batch_dev_label,False) 
            elapsed_time = time.time() - start_time
            if ( (epoch % reportEvery) == 0 ) | ( epoch==1) :
                print('DEVELOPMENT-SET total loss {} , accuracy {} , time second {} '.format(lossOnDevSet,accuracy,elapsed_time)) 
            trackingLossDev = np.append( trackingLossDev, lossOnDevSet)
            
            ## need to fix the learning_rate 
            if (trackingLossTrain[epoch] > trackingLossTrain[epoch-1]) | (trackingLossDev[epoch] > trackingLossDev[epoch-1]) : ## loss should be decreasing, if not... we need to decrease learn_rate  
                self.lr = self.lr * .8
                self.optimizer.param_groups[0]['lr'] = self.lr
            else: 
                if (epoch % 10 == 0): 
                    self.lr = self.lr * .9
                    self.optimizer.param_groups[0]['lr'] = self.lr

            # exit()
            ## https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch/49078976#49078976
            if epoch > 5: 
                torch.save( self.model, self.save_path + 'ep{}train{}dev{}.pytorch'.format(epoch, int(total_loss) , int(lossOnDevSet) ) )
        
            # print ("\n"+"-"*10+"\n")
        ##
        ##    
        print ( trackingLossTrain )
        print ( trackingLossDev )
           
    
    def get_accuracy (self, predicted_prob, true_class ): 
        values, indices = predicted_prob.max(1)
        total_correct = torch.eq ( indices,true_class) 
        total_correct1 = total_correct.cpu().data.numpy()
        return np.sum(total_correct1) / len(indices) 

