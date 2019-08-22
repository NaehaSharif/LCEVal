 # -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:22:44 2018

@author: 22161668
"""


import tensorflow as tf

from datetime import datetime 
import numpy as np
import os
import sys # 
import time
import math
import json
import matplotlib.pyplot as plt
import scipy
import random
import time
import math


import configuration # class that controls hyperparameters
from nn_classify_model import build_model, nn_out_layers, hidden_layers 
from nn_classify_utils import load_nnclassify_data, sample_minibatch,process_scores,correlation_for_validation, shuffle_data

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

FLAGS = None 

model_name='nn_classify.ckpt'

mode = 'train'

def _step_test(sess, model, features):
    
 
    nn_score= sess.run([model['nn_score']], 
                       feed_dict={model['sentence_features']: features})

    
    return nn_score   

def _validation(sess, data, model): 
    """
    Make a single gradient update for batch data. 
    """

    features= data['val_features']
    true_out= data['val_labels']



    total_loss_value,accuracy,smm,nn_score= sess.run([model['total_loss_fine'],
                                             model['accuracy_fine'],
                                             model['summaries_fine'],
                                             model['nn_score']], 
                                  feed_dict={model['sentence_features']: features,
                                             model['true_out']: true_out})

    return total_loss_value,accuracy,smm,nn_score


#------------------- the function we call to get losses-------------------------------------------------
def _train(sess, data, train_op, model,k): # returns the training loss value 
    """
    Make a single gradient update for batch data. """

    minibatch = sample_minibatch(data, model_config.train_batch_size,'train',k)
    
    features, true_out = minibatch

    _,total_loss_value,accuracy,smm,nn_score= sess.run([train_op,
                                             model['total_loss_fine'],
                                             model['accuracy_fine'],
                                             model['summaries_fine'],
                                             model['nn_score']], 
                                  feed_dict={model['sentence_features']: features,
                                             model['true_out']: true_out})
      
    return total_loss_value,accuracy,smm


#-----------Train the model------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def main(_):
 
    tf.set_random_seed(123)
    random.seed(123)

    tf.reset_default_graph() # create a new graph and start fresh
    
    g = tf.Graph()
    
    with g.as_default(): # Returns a context manager that makes this Graph the default graph.
        
        model = build_model(model_config)
        
########################################################################################################      

        """fine-tuning op""" 
        train_op = tf.contrib.layers.optimize_loss(loss=model['total_loss_fine'],
                                                   global_step=model['global_step'],
                                                   learning_rate=training_config.learning_rate,
                                                   optimizer=training_config.optimizer)
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        init = tf.global_variables_initializer()
        local= tf.local_variables_initializer()
        timestamp = str(math.trunc(time.time()))
        
        train_writer = tf.summary.FileWriter("log/" + timestamp +"training")
        validation_writer = tf.summary.FileWriter("log/" + timestamp +"validation")
        
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        with tf.device('/gpu:1'): 
            sess.run(local)
            sess.run(init)
            
           
            print ("\n training start \n")
           
            data,valIds = load_nnclassify_data(FLAGS.data_dir)
            model_config.val_batch_size=len(valIds)

            
            num_train = data['train_features'].shape[0]
            iterations_per_epoch = math.ceil(training_config.total_examples/model_config.train_batch_size) # all the training examples will be used per epoch
            num_iterations = int(training_config.total_num_epochs * iterations_per_epoch)
           
            
            # Set up some variables for book-keeping
            
            best_acc_iteration=0 # iteration number for the best accuracy
            best_acc = 0.00 # best accuracy on training set
            
            best_acc_loss=0
            best_params = {}
            loss_history = []
            acc_history=[]
            
            train_acc_history = []
            val_acc_history = []
            val_loss_history =[]
            val_correlation={}
            val_correlation_history=[]
            checkpoint_history=[]
            k=0

            print("\n\nTotal training iter: ", num_iterations, "\n\n")   #total number of iterations  
            time_now = datetime.now()
            sh_data=data
            
            shuffle=[random.sample(range(0, num_train), num_train) for i in range(training_config.total_num_epochs)]
            sh_count=0
            
            print('shuffle list created')
            data_len = len(data['train_features']-1)
            val_len=len(valIds)
            batch_size = model_config.train_batch_size
            
            for t in range(num_iterations):
                
                if(k+batch_size)< (data_len):
                    k=k+batch_size
                     
                elif(k+batch_size)> (data_len):
                    k=(data_len-batch_size)   
                else:    
                    k=0
                    sh_data=shuffle_data(data,shuffle[sh_count])
                    sh_count=sh_count+1

                #print(data_len)
                total_loss_value,acc,smm = _train(sess, sh_data, train_op, model,k) # run each training step 

                loss_history.append(total_loss_value)
                
                acc_history.append(acc)
                
        
                if FLAGS.print_every > 0 and (t +1) % FLAGS.print_every == 0:
                    
                    # Print out training loss
                    print('(Iteration %d / %d) loss: %f, and time eclipsed: %.2f minutes' % (
                            t + 1, num_iterations, float(loss_history[-1]), (datetime.now() - time_now).seconds/60.0))
                    
                    # Print out training accuracy
                    print('(Iteration %d / %d) accuracy:  %f, and time eclipsed: %.2f minutes' % (
                            t + 1, num_iterations, float(acc_history[-1]), (datetime.now() - time_now).seconds/60.0))   
                    train_writer.add_summary(smm,t)
                    
#                 Print out some correlation
                if FLAGS.sample_every > 0 and (t+1) % FLAGS.sample_every == 0:
                    
                    
                    
                    features=data['val_features'][:]
                    
                    sc = _step_test(sess,model,features)  # run each training step 
                    scores=process_scores(sc,val_len )
                    val_correlation[t]=correlation_for_validation(scores,sh_data['val_judgements'])
                    
                    val_correlation_history.append(val_correlation[t])
                    
                    print('iteration {}/{}  : pearson {}, spearman {}, kendal {}'.format(t + 1, num_iterations, 
                                                                                  val_correlation[t][0],
                                                                                  val_correlation[t][1],val_correlation[t][2]))   
#


                if FLAGS.sample_every  > 0 and (t+1) % FLAGS.sample_every  == 0:
                    
                    total_loss_value_v,vacc,vsmm,score = _validation(sess, sh_data, model) # run each training step 
                     
                    val_loss_history.append(total_loss_value_v)
                    val_acc_history.append(vacc)
                    checkpoint_history.append(t)
                    
                    print('(Iteration %d / %d) validation accuracy:  %f,validation loss:  %f, and time eclipsed: %.2f minutes' % (
                            t + 1, num_iterations, float(val_acc_history[-1]),float(val_loss_history[-1]), (datetime.now() - time_now).seconds/60.0)) 
                    validation_writer.add_summary(vsmm,t)
                
                    if (vacc>best_acc):
                        best_acc=vacc
                        best_acc_loss=total_loss_value_v
                        best_acc_iteration=t
                        


                if FLAGS.saveModel_every > 0 and (t+1) % FLAGS.saveModel_every == 0:
                    if not os.path.exists(FLAGS.savedSession_dir):
                        os.makedirs(FLAGS.savedSession_dir)
                    checkpoint_name = model_name[:-5] + '_checkpoint{}.ckpt'.format(t+1)
                    save_path = model['saver'].save(sess, os.path.join(FLAGS.savedSession_dir, checkpoint_name))
                        

            save_path = model['saver'].save(sess, os.path.join(FLAGS.savedSession_dir, model_name))
            print("done. Model saved at: ", os.path.join(FLAGS.savedSession_dir, model_name))  
            print('(Iteration %d ) best_acc: %f,loss: %f, and total_time eclipsed: %.2f minutes' % (
                            best_acc_iteration,float(best_acc), float(best_acc_loss), (datetime.now() - time_now).seconds/60.0))
            
            

        sess.close()


#------------this code takes command line arguments to set the FlAGS---------------------------------------------            
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--savedSession_dir',
      type=str,
      default='D:/NNeval_classification/nnclassify_session_'+model_config.case+'/',
      help="""\
      Directory where your created pretrained model session will be saved.
      """
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='D:/NNeval_classification/data/',
      help='Directory where all your training and validation data in json file can be found.'
  )
  parser.add_argument(
      '--print_every',
      type=int,
#      default=2040,
      default=math.ceil(training_config.total_examples/model_config.train_batch_size),
      help='Num of steps to print your training loss. 0 for not printing/'
  )
  parser.add_argument(
      '--sample_every',
      type=int,
      default=math.ceil(training_config.total_examples/model_config.train_batch_size),
      help='Num of steps to generate validation resullts for training.  0 for not validating.'
  )
  parser.add_argument(
      '--saveModel_every',
      type=int,
      default=math.ceil((training_config.total_examples/model_config.train_batch_size)/training_config.models_to_save_per_epoch), # one epoch has 3058 iterations  , for batch 30 and examples 91744
      help='Num of steps to save model checkpoint for trainig. 0 for not doing so.'
  )
  parser.add_argument(
      '--val_correlation_every',
      type=int,
      #default=2040,
      default=math.ceil(training_config.total_examples/model_config.train_batch_size), 
      help='Num of steps to check correlation for validation data. 0 for not doing so.'
  )
  parser.add_argument(
      '--val_judgement_path',
      type=str,
      default='D:/NNeval_classification/data/val/val_judgements.json',
      help='path of the validation judgements.'
  )
  parser.add_argument(
      '--val_path',
      type=str,
      default='D:/NNeval_classification/data/val/val_features.json',
      help='path of the validation features.'
  )


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
