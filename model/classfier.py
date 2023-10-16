import tensorflow as tf
import keras
import keras.layers as layers
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50,ResNet101,ResNet152
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2

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
        


base = InceptionV3(include_top=False, weights='imagenet', 
                   input_shape=(para.size_holder[0],para.size_holder[1],3))        

class Inception_V3(Model):
    def __init__(self,feature_out=False,vector_out=False,dim_out=para.classNums):
        self.feature_out = feature_out
        self.vector_out = vector_out
        self.dim_out = dim_out
        super(Inception_V3, self).__init__()

        self.base_out = Model(inputs=base.input,outputs=[base.output,base.get_layer(name='conv2d_89').output],name="inception_v3")
        self.gpa = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(1024)
        self.relu_1 = layers.ReLU()
        self.pred = layers.Dense(self.dim_out,activation='softmax')
        
    def call(self, x):

        x,f = self.base_out(x)
        x = self.gpa(x)
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


        
class ResNet_101(Model):
    def __init__(self,feature_out=False,vector_out=False,dim_out=para.classNums,pretrain_load='imagenet'
                 ,input_shape=(para.size_holder[0],para.size_holder[1],3)):
        self.feature_out = feature_out
        self.vector_out = vector_out
        self.dim_out = dim_out
        super(ResNet_101, self).__init__()
        self.feature_extractor = ResNet101(include_top=False, weights=pretrain_load, 
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
        

class ResNet_152(Model):
    def __init__(self,feature_out=False,vector_out=False,dim_out=para.classNums,pretrain_load='imagenet'
                 ,input_shape=(para.size_holder[0],para.size_holder[1],3)):
        self.feature_out = feature_out
        self.vector_out = vector_out
        self.dim_out = dim_out
        super(ResNet_152, self).__init__()
        self.feature_extractor = ResNet152(include_top=False, weights=pretrain_load, 
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
 

class MobileNet_v1(Model):
    def __init__(self,feature_out=False,vector_out=False,dim_out=para.classNums,pretrain_load='imagenet'
                 ,input_shape=(para.size_holder[0],para.size_holder[1],3)):
        self.feature_out = feature_out
        self.vector_out = vector_out
        self.dim_out = dim_out
        super(MobileNet_v1, self).__init__()
        self.feature_extractor = MobileNet(include_top=False, weights=pretrain_load, 
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
        

class MobileNet_v2(Model):
    def __init__(self,feature_out=False,vector_out=False,dim_out=para.classNums,pretrain_load='imagenet'
                 ,input_shape=(para.size_holder[0],para.size_holder[1],3)):
        self.feature_out = feature_out
        self.vector_out = vector_out
        self.dim_out = dim_out
        super(MobileNet_v2, self).__init__()
        self.feature_extractor = MobileNetV2(include_top=False, weights=pretrain_load, 
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