import tensorflow as tf
import keras as keras
import keras.layers as layers
from keras import Model

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50

import utils.para as para
import numpy as np



class VGG_16(Model):
    def __init__(self,feature_out=False,vector_out=False,dim_out=para.classNums,pretrain_load='imagenet'
                 ,input_shape=(para.size_holder[0],para.size_holder[1],3)):
        self.feature_out = feature_out
        self.vector_out = vector_out
        self.dim_out = dim_out
        super(VGG_16, self).__init__()
        self.feature_extractor = VGG16(include_top=False, weights=pretrain_load, 
                   input_shape=input_shape)
        # self.fat = layers.Flatten()
        # self.gpm = layers.GlobalMaxPool2D()
        self.gpa = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(1024)
        self.relu_1 = layers.ReLU()
        # self.drop = layers.Dropout(0.5)
        # self.lkrelu_1 = layers.LeakyReLU(0.1)
        self.pred = layers.Dense(self.dim_out,activation='softmax')
        
    def call(self, x):
        f = self.feature_extractor(x)
        x = self.gpa(f)
        v = self.dense_1(x)
        x = self.relu_1(v)
        x = self.pred(x)
        
        if (self.feature_out==True and self.vector_out==True):
            return x,f,v
        elif (self.feature_out==True):
            return x,f
        elif (self.vector_out==True):
            return x,v
        else:
            return x
        

class ResNet_50(Model):
    def __init__(self,feature_out=False,vector_out = False,dim_out=para.classNums,pretrain_load='imagenet'
                 ,input_shape=(para.size_holder[0],para.size_holder[1],3)):
        self.feature_out = feature_out
        self.vector_out = vector_out
        self.dim_out = dim_out
        super(ResNet_50, self).__init__()
        self.feature_extractor = ResNet50(include_top=False, weights=pretrain_load, 
                   input_shape=input_shape)
        self.gpa = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(1024)
        self.relu_1 = layers.ReLU()
        self.pred = layers.Dense(self.dim_out,activation='softmax')
        
    def call(self, x):
        f = self.feature_extractor(x)
        x = self.gpa(f)
        v = self.dense_1(x)
        x = self.relu_1(v)
        x = self.pred(x)
          
        if (self.feature_out==True and self.vector_out==True):
            return x,f,v
        elif (self.feature_out==True):
            return x,f
        elif (self.vector_out==True):
            return x,v
        else:
            return x