from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import pickle
import sys
import time
import copy
import numpy as np

import Prediction_model.classification_model as models
#from deep_fairness.fairyted import Fairytale
from Prediction_model.helper import load_pickle, sample_indices, cvt, make_minibatch, counterfactual_loss, calc_acc, MacOSFile, variation_loss

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

import torch
import torch.nn as nn
import math
import itertools
import random
import pickle
import numpy as np
import torch.optim as optim




class Experiment_diversifier(object):

  def __init__(self, config):
    super(Experiment_diversifier, self).__init__()
    self.config = config

    # assert(self.config['model_params']['in_size'] == self.config['trans_dim'] + 
    #   self.config['u_dim'] + self.config['view_dim'] + self.config['a_dim'])

    # Use appropritate device
    if self.config['gpu_num'] < 0:
      self.device = torch.device('cpu')
    else:
      self.device = torch.device('gpu:{0}'.format(self.config['gpu_num']))

    
    data_dict_path = os.path.join(self.config['input_path'], self.config['input_filename'])
    data_dict = load_pickle(data_dict_path)
    for key in data_dict:
      data_dict[key] = np.array(data_dict[key])
    self.data = data_dict
    # if self.config['use_simulated_data']:
    #   self.fairmodel = Fairytale()
    # else:
    #   self.fairmodel = Fairytale(data=data_dict)

    # Prepare the neural network and others
    self.total_epoch = 0
    self.best_loss = float('inf')
    self.model = getattr(models, self.config['model_type'])(**self.config['model_params']).to(device=self.device)
    self.loss_fn = getattr(nn, self.config['loss_function_name'])()
    self.optimizer = getattr(optim, self.config['optimizer'])(self.model.parameters(),**self.config['optimizer_params'])
    self.scheduler = getattr(optim.lr_scheduler, self.config['scheduler'])(self.optimizer, **self.config['scheduler_params'])
    self.relu = nn.ReLU()
    #self.cf_loss = counterfactual_loss
    self.text_divloss = variation_loss

  def set_random_seed(self):      
    seed = self.config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




  def test_model(self, orig_data, test_idx):
    self.model.eval()

    
    inputs = orig_data['input'][test_idx,:]
    labels = orig_data['label'][test_idx,:]
    #print(inputs)
    with torch.set_grad_enabled(False):
      outputs = self.model(inputs)
      x,total_acc = calc_acc(outputs, labels)
    #print(outputs)
    average_test_acc = np.mean(total_acc.numpy(),axis=0)
    print("Test Accuracy is :",average_test_acc)
    print("Model Accuracy: ",np.mean(average_test_acc))
    return x,average_test_acc


  def train_model(self, orig_data, train_idx, dev_idx, 
    max_epochs=10, max_iter=10, use_textdiv=False, minibatch_size=10):


    
    since = time.time()

    best_model_wts = copy.deepcopy(self.model.state_dict())
    text_divloss_list = []

    for epoch in range(max_epochs):
      
      print('Epoch {}/{}'.format(epoch, max_epochs - 1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'val']:
        if phase == 'train':
          self.model.train()  # Set model to training mode
          indices = train_idx
        else:
          self.model.eval()   # Set model to evaluate mode
          indices = dev_idx

        running_loss = 0.0
        text_divloss = 0
        # Iterate over data.
        for iter, a_batch in enumerate(make_minibatch(indices, minibatch_size)):
          
          if iter > max_iter:
            break

          inputs = orig_data['input'][a_batch,:]
          if self.config['use_binned_text_diversity']:
            text_diversity = orig_data['binned_text_diversity'][a_batch]
          else:
            text_diversity = orig_data['text_diversity'][a_batch]
          labels = orig_data['label'][a_batch,:]


          # zero the parameter gradients
          self.optimizer.zero_grad()

          # forward
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            # outputs.register_hook(lambda grad: print("loss grad",grad))

            #loss.register_hook(lambda: grad)
            

            # backward + optimize only if in training phase
            if phase == 'train':
                if use_textdiv:
                  #print("hooo=====")
                  # print()
                  #print(text_diversity)
                  lamb = 1
                  divloss_list = self.text_divloss(outputs,labels,text_diversity)
                  temp_tdloss = torch.mean(self.relu(divloss_list))
                  # temp_tdloss.register_hook(lambda grad: print("div loss grad ",grad))
                  #print(loss,'before')
                  loss = loss+ lamb*temp_tdloss
                  #print(loss,'after')
                  text_divloss = text_divloss + lamb*temp_tdloss
                loss.backward()
                self.optimizer.step()                
                
          # statistics
          running_loss += loss.item()
        if phase=='train':
          text_divloss_list.append(text_divloss)

          

        epoch_loss = running_loss / (max_iter * minibatch_size)
        epoch_text_divloss = text_divloss/ (max_iter)
        print('{} Loss: {:.4f} '.format(phase, epoch_loss))
        if use_textdiv:
          print('{} Loss: {:.4f} '.format('textdiv', epoch_text_divloss))

        # deep copy the model
        if phase == 'val' and epoch_loss < self.best_loss:
          self.best_loss = epoch_loss
          best_model_wts = copy.deepcopy(self.model.state_dict())

      

    time_elapsed = time.time() - since

    self.total_epoch += epoch 
    print('Training complete in {:.0f}m {:.0f}s (total epoch = {})'.format(
      time_elapsed // 60, time_elapsed % 60, self.total_epoch))

    # load best model weights
    self.model.load_state_dict(best_model_wts)
    with open('Output/loss_fn.pkl','wb') as f:
      pickle.dump(text_divloss_list,f)

  def save_model(self, model_filepath):
    checkpoint = {
      'total_epoch':self.total_epoch,
      'model_state_dict':self.model.state_dict(),
      'optimizer_state_dict':self.optimizer.state_dict(),
      'scheduler_state_dict':self.scheduler.state_dict(),
      'best_loss':self.best_loss
    }
    torch.save(checkpoint, model_filepath)
    

  def load_model(self, model_filepath):
    checkpoint = torch.load(model_filepath)
    self.total_epoch = checkpoint['total_epoch']
    self.best_loss = checkpoint['best_loss']
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  def split(self,num_inp):
    
    # Divide into train/dev/test
    # ==========================
    print("splitting in process")
    train_idx, dev_idx, test_idx = sample_indices(num_inp)
    index_filename = os.path.join(self.config['output_path'], self.config['split_filename'])
    pickle.dump((train_idx, dev_idx, test_idx), open(index_filename, 'wb'))
    return train_idx, dev_idx, test_idx

  def load_split_index(self):
    index_filename = os.path.join(self.config['output_path'], self.config['split_filename'])
    train_idx, dev_idx, test_idx = pickle.load(open(index_filename, 'rb'))
    return train_idx, dev_idx, test_idx


  def run(self):
    '''
    Runs the experiment
    '''
    orig_data = {}
    if self.config["input_type"]=='transcript_only':
    	orig_data["input"] = self.data['transcript']
    	orig_data["label"] = self.data['rating']
    	orig_data["text_diversity"] = self.data['text_diversity']
 
     
    # Divide into train/dev/test
    # ==========================
    if self.config["need_split"]:
      train_idx, dev_idx, test_idx = self.split(orig_data['input'].shape[0])
      
    else:
      train_idx, dev_idx, test_idx = self.load_split_index()

    # Binned diversity calculation
    # ==========================
    self.min_train_div = min(self.data['text_diversity'][train_idx])
    self.max_train_div = max(self.data['text_diversity'][train_idx])
    diff = self.max_train_div - self.min_train_div
    bin_len = diff/self.config['num_bin']
    binned_text_diversity = np.array([int(x/bin_len)*bin_len+bin_len/2.0 for x in orig_data['text_diversity']])
    orig_data['binned_text_diversity'] = binned_text_diversity
    print("Splitting done")
    

    # Send to appropriate device
    # ==========================
    for a_key in orig_data:
      orig_data[a_key] = torch.from_numpy(orig_data[a_key].astype(np.float32)).to(device=self.device)
    print("Data Load done") 



    # Neural network training part
    # ============================
    if self.config['train_neural_network']:
      #print("Training Model number: ",model_id)
      nn_filename = os.path.join(self.config['output_path'], self.config['neural_network_model_filename'])
      if self.config['load_nn_from_file']:
        self.load_model(nn_filename)
      self.train_model(orig_data, train_idx, dev_idx, **self.config['trainer_params'])
      self.save_model(nn_filename)

    # Test Neural Network
    # ===================
    if self.config['test_neural_network']:
      nn_filename = os.path.join(self.config['output_path'], self.config['load_nn_filename'])
      self.load_model(nn_filename)
      op,average_test_acc = self.test_model(orig_data, test_idx)
      op = op.numpy()
      inp =  orig_data['input'][test_idx,:].numpy()
      data_dict_predict, data_dict_true = {},{}
      data_dict_true['transcript'] = inp[:,:200]
      data_dict_predict['transcript'] = inp[:,:200]

      # data_dict_true['a'] = inp[:,200:207]
      # data_dict_true['view'] = inp[:,207]
      # data_dict_true['rating'] = orig_concat_data['label'][test_idx,:].numpy()
      
      # data_dict_predict['a'] = inp[:,200:207]
      # data_dict_predict['view'] = inp[:,207]
      
      data_dict_true['rating'] = orig_data['label'][test_idx,:].numpy()
      data_dict_predict['rating'] = op
      self.data_dict_predict = data_dict_predict
      self.data_dict_true = data_dict_true 
      with open('Output/test_output_'+self.config["input_type"]+'.pkl','wb') as f:
        pickle.dump((data_dict_predict,data_dict_true),f) 


    


