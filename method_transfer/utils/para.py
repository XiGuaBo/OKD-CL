import numpy as np
import tensorflow as tf

# dataset config

train_data = "/home/temp/data-train-15/train"

test_data = "/home/temp/data-train-15/test"

hard_mask = "/home/temp/data-train-15/Hard_masks"


classNums = 15


lab_type = 'Hard_masks' 

# train config

    # pictures preprocess config
    
size_holder = (224,224)

train_resize = True

batch_size = 1

epochs = 100

base_linear_epochs = 20

base_opt_convlast_epochs = 10

base_opt_convall_epochs = 10

constraint_epochs = 20

    # lr config
base_linear_lr = 1e-4

base_opt_convlast_lr = 1e-5

base_opt_convall_lr = 1e-6

fine_tune_lr = 1e-5

fine_tune_batch_size = 1

class lrSch(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, lr):
    self.lr = lr
  def __call__(self, step):
    return self.lr*(np.power(0.99,step))

def lr_decay(lr,epochs):
    # loss function
    return tf.math.pow(0.99,epochs)*lr

def lr_decay_specific(lr,epochs,decay_rate):
    # loss function
    return tf.math.pow(decay_rate,epochs)*lr

def fgAttentionLoss(y_true,y_pred):
    fgpixels = tf.reduce_sum(y_true*y_pred)
    # bgpixels = tf.reduce_sum(y_pred)-fgpixels
    return 1 - (fgpixels/(tf.reduce_sum(y_pred)+1e-4))

def mseLoss(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def maeLoss(y_true,y_pred):
    return tf.reduce_mean(tf.math.abs(y_true-y_pred))

def categoryHingeLoss(y_true,y_pred):
    return tf.math.maximum(tf.math.maximum((1-y_true)*y_pred) - tf.reduce_sum(y_true*y_pred) + 1,0)

def categoricalCrossentropyLoss(y_true,y_pred):
    return tf.math.maximum(tf.reduce_sum(y_true*(-tf.math.log(tf.math.minimum(tf.math.maximum(y_pred,0),1-1e-7)+1e-7))),0)

def attention_weights_produce(epoch,weights_init=1e-4):
    return weights_init*(tf.math.pow(1.05,epoch))

# test config

test_resize = False

# heatmap export config

keep_percent = 30

# Debug

debug = True

check_idx_range = [623,633]