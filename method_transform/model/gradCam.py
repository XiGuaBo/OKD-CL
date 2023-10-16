import tensorflow as tf
import keras as keras
import keras.layers as layers
from keras import Model
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_heatmap(GradCamMod, x, classIdx, upsample_size, eps=1e-5, tape=None, training=True):
        # record operations for automatic differentiation
        (preds ,convOuts) = GradCamMod(x,training=training)  # preds after softmax
        loss = preds[:, classIdx]
        # compute gradients with automatic differentiation
        if (tape==None):
            print ("Please Transmit Tape First!")
            exit(-1)
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1, keepdims=True)
        cam = tf.math.maximum(cam,0)
        if (tf.reduce_max(cam)!=0):
            cam = cam / (tf.reduce_max(cam))
        # cam = tf.image.resize(cam,upsample_size)

        cam_rsz = cv2.resize(cam.numpy().squeeze(axis=-1),upsample_size)
        # convert to 3D
        cam3 = np.expand_dims(cam_rsz, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        return cam3, cam, preds
    
def overlay_gradCAM(img, cam3, cam):
    new_img = cam * img * 255
    new_img = new_img.astype("uint8")

    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    new_img_concat = 0.3 * cam3 + 0.5 * img * 255.0
    new_img_concat = (new_img_concat / new_img_concat.max() * 255.0).astype("uint8")

    return new_img, new_img_concat

def GradResShow(img,hm,seg):
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor((img.squeeze(axis=0)*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    plt.title("org")
    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(hm,cv2.COLOR_RGB2BGR))
    plt.title("heatmap")
    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(seg,cv2.COLOR_RGB2BGR))
    plt.title("seg")
    plt.show()
