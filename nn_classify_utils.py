# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:29:54 2018
utils for nn_classify 
@author: 22161668
"""

import json
import numpy as np
import os
import numpy as np
import h5py
from collections import Counter
import random
import time
import scipy
from datetime import datetime



def load_nnclassify_data(base_dir='D:/LCEval_classification/data'):
    
  data = {}
  start_t = time.time()
  
  train_feature_file = os.path.join(base_dir, 'train/train_features.json')
  train_labels_file = os.path.join(base_dir, 'train/train_labels.json')
  train_imageid_file = os.path.join(base_dir, 'train/train_imageid.json')
  
  val_feature_file = os.path.join(base_dir, 'val/val_features.json')
  val_labels_file = os.path.join(base_dir, 'val/val_labels.json')
  val_imageid_file = os.path.join(base_dir, 'val/val_imageid.json')
  val_judgements_file = os.path.join(base_dir, 'val/val_judgements.json')
  
  with open(train_feature_file) as f:  # loading training features
    train_feats=json.load(f)
    
  with open(val_feature_file) as f: # loading validation features
    val_feats=json.load(f)    
    
    
  trainIds=train_feats.keys()
  valIds=val_feats.keys()
 
  
  
  with open(train_labels_file) as f:  # loading training labels
    train_lb=json.load(f)       
    
  with open(val_labels_file) as f: # loading validation labels
    val_lb=json.load(f)    
    
    
  with open(train_imageid_file) as f: 
    train_imd=json.load(f)
    
  with open(val_imageid_file) as f:
    val_imd=json.load(f)   
    
  with open(val_judgements_file) as f:
    val_jdg=json.load(f)   
    
  train_features=[]
  val_features=[]
  train_labels=[]
  val_labels=[]
  train_imageid=[]
  val_imageid=[]
  val_judgements=[]
  
    
  for i in trainIds:
       train_features.append(train_feats[i])
      
       train_labels.append(int(train_lb[i]))
       
       train_imageid.append(train_imd[i])
       
       
  for j in valIds:
      
       val_features.append(val_feats[j])
       
       val_labels.append(int(val_lb[j]))
       
       val_imageid.append(val_imd[j])
       
       val_judgements.append(val_jdg[j])
       
       
  data['train_features']=np.array(train_features)
  data['val_features']=np.array(val_features)
   
  data['train_labels']=np.array(train_labels)
  data['val_labels']=np.array(val_labels)
   
  data['train_imageid']=np.array(train_imageid)
  data['val_imageid']=np.array(val_imageid)
  
  data['val_judgements']=np.array(val_judgements)
  
  end_t = time.time()
  print ("Data loading Elapse time: %.2f" %(end_t - start_t))


  return data, valIds


def sample_minibatch(data, batch_size, split,k):
  

  labels = data['%s_labels' % split][k:k+batch_size]
  sentence_features = data['%s_features' % split][k:k+batch_size]
  

  return sentence_features, labels


def load_test_data(test):
    
  data = {}
  start_t = time.time()
  
  train_feature_file =test
  
  
  with open(train_feature_file) as f:  # loading training features
    test_feats=json.load(f)
 
      
  testIds=test_feats.keys()

  test_features=[]
  ids=[]
    
  for i in testIds:
       
      test_features.append(test_feats[i])
       
      ids.append(i)
            
  data['test_features']=np.array(test_features)
  
  end_t = time.time()
  #print ("Data loading Elapse time: %.2f" %(end_t - start_t))

  return data,ids


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def correlation_for_validation(scores,judgement):
        
    j_all=judgement
    scores=np.array(scores)
    
    pearson=[]
    spearman=[]
    kendal_tau=[]

    pearson.append(np.corrcoef([j_all,scores])[0, 1])
    
    spr,_=scipy.stats.spearmanr(j_all,scores)
    spearman.append(spr)
    
    tau,_= scipy.stats.kendalltau(j_all,scores)
    kendal_tau.append(tau)
    
    return [pearson[0],spearman[0],kendal_tau[0]]
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def process_scores(sc,BATCH_SIZE_INFERENCE):

    score_list=[]
    
    
    for i in range(BATCH_SIZE_INFERENCE):
        score_list.append(sc[0][i][1])
    
    return score_list
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def shuffle_data(data,shuffle):

    data['train_features']=data['train_features'][shuffle]
    data['val_features']=data['val_features']
       
    data['train_labels']=data['train_labels'][shuffle]
    data['val_labels']=data['val_labels']
       
    data['train_imageid']=data['train_imageid'][shuffle]
    data['val_imageid']=data['val_imageid']
#      
    data['val_judgements']=data['val_judgements']
    #print('Shuffle data called at {}:{}'.format(((datetime.now() - time_now).seconds),(datetime.now() - time_now).seconds/60))
            
    return data


