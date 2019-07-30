# -*- coding: utf-8 -*-
"""
@author: 22161668
"""
# classification configuration


"""LCEval model and training configurations."""

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""

    
    self.val_batch_size = 3976 # size of the validation set 

    self.sentence_feature_size = 16
    self.nneval_insize= 16
    
    self.classes = 2
    
    self.initializer_scale = 0.001
    

    self.train_batch_size =13300
    self.layer1out_size =12 #
    self.layer2out_size = 8 #
    self.case='final-NNEVAL-all_5'

class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    self.learning_rate = 0.0005
    
    
    self.total_examples =132984#155148#177312#88656#132984
    #3058 interations per epoch
    # Optimizer for training the model.
    self.optimizer = "Adam"

    self.models_to_save_per_epoch = 1
    self.total_num_epochs = 1000
    
