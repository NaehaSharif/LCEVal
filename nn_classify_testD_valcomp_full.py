# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:45:06 2018

@author: 22161668
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:28:23 2018
testing nneval_classify and correlation evaluation
@author: 22161668
"""

# -*- coding: utf-8 -*-

""" generate scores of the test data""" # classification
""" funcions"""
""" load_compare_data"""  # returns captions and features of the whole test data
""" _step_test"""  # generates scores for the input captions and feature pairs
""" save_result""" # saves the generated scores into a csv file

# nneval_pretrain test

import os, json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import scipy.stats
import math

import configuration # class that controls hyperparameters
from nn_classify_model import* # makes the main graph # its the backbone 
from nn_classify_utils import load_test_data

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

verbose = True
f=math.ceil((training_config.total_examples/model_config.train_batch_size)/training_config.models_to_save_per_epoch)
number=training_config.models_to_save_per_epoch*f*training_config.total_num_epochs
directory='D:/NNeval_classification/nnclassify_session_'+model_config.case+'/'
#directory='D:/backup/progress_update/model 22/nnclassify_session/'
modelname='nn_classify_checkpoint{}.ckpt'.format(number)

MAX=math.ceil(number/f)

checking=[x * f for x in range(2,int(MAX))]
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def _step_test(sess, model, features):
    
 
    nn_score= sess.run([model['nn_score']], 
                       feed_dict={model['sentence_features']: features})

    
    return nn_score   
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def process_scores(sc,BATCH_SIZE_INFERENCE):

    score_list=[]
    
    
    for i in range(BATCH_SIZE_INFERENCE):
        score_list.append(sc[0][i][1])
    
    return score_list
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
def get_accuracy(scores, labels_path, Ids):
    
    data_dir=labels_path
    
    with open(data_dir) as f:
         judgements=json.load(f)
        
    j_all=[]
 
    
    for i in Ids: 
        j_all.append(judgements[i])
        
    
    human=0
    machine=0
    count=0.0
    
    for i,sc in enumerate(scores):
        count=count+1
        if (sc>=0.5):
            if (j_all[i]==1):
                human=human+1
        else:
            if (j_all[i]==0):
                machine=machine+1
                
    accuracy=(human+machine)/count
    
    return accuracy            
                

    
    
def get_correlation(scores, judgement_path, Ids):
    # getting correlation with flicker 8k scores 
    data_dir=judgement_path
    
    # val_scores.json       {0: 1, 1:5}
    with open(data_dir) as f:
         judgements=json.load(f)
        
    j_all=[]
   
    for i in Ids: 
        j_all.append(judgements[i])
        
    j_all=np.array(j_all)
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
    
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 


base_dir = 'D:/NNeval_classification/data'
test=os.path.join(base_dir,'valcomp_full/valcomp_full_features.json')
features_dict, Ids = load_test_data(test)
features=features_dict['test_features'][:]
judgement='D:/NNeval_classification/data/valcomp_full/valcomp_full_judgements.json'
label='D:/NNeval_classification/data/valcomp_full/valcomp_full_labels.json'
TOTAL_INFERENCE_STEP = 1
BATCH_SIZE_INFERENCE = len(features)
pearson_COMP=[]
spearman_COMP=[]
kendal_COMP=[]
validation_accu=[]

#-----------------------------------
mean_COMP=[]
pearsonAVG=[]
spearmanAVG=[]
kendalAVG=[]
#----------------------------
print('Total test examples :{}'.format(BATCH_SIZE_INFERENCE))
# Build the TensorFlow graph and train it


g = tf.Graph()
with g.as_default():
    # Build the model.

    model = build_model(model_config)
    
    print('graph loaded!')
    
    # run training 
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    
    #checking=[1110]
    with tf.device('/gpu:1'): 
    
        sess.run(init)
        for s in checking:
            
            model['saver'].restore(sess, os.path.join(directory,'nn_classify_checkpoint{}.ckpt'.format(s)))
              
            #print("Model restured! Last step run: ", sess.run(model['global_step']))
            
            for i in range(TOTAL_INFERENCE_STEP):
               
                sc = _step_test(sess,model,features) # the output is size (32, 16)
                
#            print("processing_successful")
            scores=process_scores(sc, BATCH_SIZE_INFERENCE)
            
#            print(scores[0:5])
            p_corr,sr_corr,tau_corr=get_correlation(scores,judgement,Ids)
            accuracy=get_accuracy(scores,label,Ids)
            
            pearson_COMP.append(p_corr)
            spearman_COMP.append(sr_corr)
            kendal_COMP.append(tau_corr)
            validation_accu.append(accuracy)
            mean_COMP.append((p_corr+sr_corr+tau_corr)/3)
            
            
#            print("inference_successful")
        #________________________________________
        N=20
        pearsonAVG=running_mean(pearson_COMP, N)
        spearmanAVG=running_mean(spearman_COMP, N)
        kendalAVG=running_mean(kendal_COMP, N)   
        #----------------------------------
        
        
        for i,check in enumerate(checking):
            checking[i]=check/f
            
        plt.plot(pearsonAVG, 'r.',
        spearmanAVG,'b.', kendalAVG, 'g.')
            
        plt.xlabel('epochs')
        plt.ylabel('correlation average')
        plt.show()  
        
        plt.plot(checking, pearson_COMP, 'r.')
            
        plt.xlabel('epochs')
        plt.ylabel('correlation')
        plt.title('pearson')
        plt.show() 
        
        plt.plot(checking, spearman_COMP, 'b.')
            
        plt.xlabel('epochs')
        plt.ylabel('correlation')
        plt.title('spearman')
        plt.show()  
        
        plt.plot(checking, kendal_COMP, 'c.')
            
        plt.xlabel('epochs')
        plt.ylabel('correlation')
        plt.title('kendal')
        plt.show()
        
        plt.plot(checking, mean_COMP, 'm.')
            
        plt.xlabel('epochs')
        plt.ylabel('correlation')
        plt.title('mean')
        plt.show()
        
        
        best_p_value_COMP=max(pearson_COMP) 
        best_p_value_COMP = format(best_p_value_COMP, '.3f')
        print( 'pearson: {}'.format(best_p_value_COMP))
        best_pearson_COMP=np.argmax(pearson_COMP)
        
        best_model_pearson_COMP=checking[best_pearson_COMP]
        print('{}'.format(best_model_pearson_COMP*f))
        #-------------------------------------------------------------
        
        best_s_value_COMP=max(spearman_COMP)  
        best_s_value_COMP = format(best_s_value_COMP, '.3f')
        print( 'spearman value: {}'.format(best_s_value_COMP))
        best_spearman_COMP=np.argmax(spearman_COMP)
        
        best_model_spearman_COMP=checking[best_spearman_COMP]
        print('{}'.format(best_model_spearman_COMP*f))
        #--------------------------------------------------------------
        
        best_kendal_value_COMP=max(kendal_COMP)
        best_kendal_value_COMP = format(best_kendal_value_COMP, '.3f')
        print( 'kendal-tau value: {}'.format(best_kendal_value_COMP))
        best_kendal_COMP=np.argmax(kendal_COMP)
        
        best_model_kendal_COMP=checking[best_kendal_COMP]
        print('{}'.format(best_model_kendal_COMP*f))
         #--------------------------------------------------------------
        best_mean_value_COMP=max(mean_COMP) 
        best_mean_value_COMP = format(best_mean_value_COMP, '.3f')
#        print( 'B mean value VAL: {}'.format(best_mean_value))
        best_mean_COMP=np.argmax(mean_COMP)
        
        best_model_mean_COMP=checking[best_mean_COMP]
        print('VAL_COMP_FULL model : {}'.format(best_model_mean_COMP*f))
        
        sess.close()
    
            
            