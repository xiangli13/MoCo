import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import time

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.activations import softmax
class MoCo(Model):
    """Momentum Contrastive Feature Learning"""
    def __init__(self, m=0.1, img_shape=(224, 224, 3), queue_len=128, feature_dim=128):
        super(MoCo, self).__init__(dynamic=True)
        self.m = m
        self.img_shape = img_shape
        self.feature_dim = feature_dim
        self.queue = tf.zeros((1, self.feature_dim))
        self.queue_len = queue_len
        self.q_enc = Sequential([
            ResNet50(include_top=False, weights=None, input_shape=self.img_shape),
            Flatten(),
            # Dense(self.feature_dim)
        ]) 
        self.k_enc = Sequential([
            ResNet50(include_top=False, weights=None, input_shape=self.img_shape),
            Flatten(),
            # Dense(self.feature_dim)
        ]) 
        self.g = Dense(self.feature_dim)
        self.k_enc.set_weights(self.q_enc.get_weights())
        
        for layer in self.k_enc.layers:
            layer.trainable = False
        
    def call(self, inputs):
        # update key extractor weights
        k_weights = [self.m*w_k + (1-self.m)*w_q for w_k, w_q \
                     in zip(self.k_enc.get_weights(), self.q_enc.get_weights())]
        self.k_enc.set_weights(k_weights)
        # get two versions of same batch data
        x_q, x_k = self.rand_aug(inputs)
        
        # save the key and query
        # sample_q = tf.io.encode_jpeg(tf.cast(x_q[0]*255, tf.uint8))
        # ts = ''.join(map(str, list(time.localtime())))
        # tf.io.write_file(f'{ts}_sample_q.jpeg', sample_q)
        # sample_k = tf.io.encode_jpeg(tf.cast(x_k[0]*255, tf.uint8))
        # tf.io.write_file(f'{ts}_sample_k.jpeg', sample_k)

        # forward
        q = self.g(self.q_enc(x_q))
        q = tf.reshape(q, (tf.shape(q)[0], 1, -1))
        k = self.g(self.k_enc(x_k))
        k = tf.reshape(k, (tf.shape(k)[0], -1, 1))
        l_pos = tf.squeeze(tf.matmul(q, k), axis=-1)
        l_neg = tf.matmul(tf.squeeze(q), tf.transpose(self.queue))
        # logits = softmax(tf.concat([l_pos, l_neg], axis=1))
        logits = tf.concat([l_pos, l_neg], axis=1)
        self.queue_them(tf.squeeze(k))
        ###### keras-fashion version ######
        # return logits
        ###### gradient-tape version ###### 
        labels = tf.zeros(tf.shape(inputs)[0])
        loss = K.mean(sparse_categorical_crossentropy(labels, logits, from_logits=True))
        l2 = tf.reduce_mean(tf.math.l2_normalize(q))
        # print(K.max(logits, axis=1).numpy())
        hits = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, 'int64'))
        acc = tf.reduce_mean(tf.cast(hits, 'float64'))
        return loss + 0.1 * l2, acc
    
    def queue_them(self, k):
        if self.queue == None:
            self.queue = k
        elif len(self.queue) >= self.queue_len:
            batch_len = tf.shape(k)[0]
            self.queue = self.queue[batch_len:]
            self.queue = tf.concat([self.queue, k], axis=0)
        else:
            self.queue = tf.concat([self.queue, k], axis=0)
    
    def rand_aug(self, batch, 
                 resize_min=1, 
                 resize_max=2, 
                 jitter_delta=0.5):
        batch_shape = tf.shape(batch)
        # random reszie
        img_size = batch_shape[1:-1].numpy()
        resize_delta = tf.random.uniform([2, 1], resize_min, resize_max).numpy()
                            
        x_k = tf.image.resize(batch, size=img_size*resize_delta[0])
        x_q = tf.image.resize(batch, size=img_size*resize_delta[1])
        # random crop
        crop_size = (*batch_shape.numpy()[:1], *self.img_shape)
        x_k = tf.image.random_crop(x_k, size=crop_size)
        x_q = tf.image.random_crop(x_q, size=crop_size)
        # random jitter
        x_k = tf.image.random_brightness(x_k, jitter_delta)
        x_k = tf.image.random_contrast(x_k, 1-jitter_delta, 1+jitter_delta)
        x_k = tf.image.random_saturation(x_k, 1-jitter_delta, 1+jitter_delta)
        x_q = tf.image.random_brightness(x_q, jitter_delta)
        x_q = tf.image.random_contrast(x_q, 1-jitter_delta, 1+jitter_delta)
        x_q = tf.image.random_saturation(x_q, 1-jitter_delta, 1+jitter_delta)
        # random horizontal flip
        x_k = tf.image.random_flip_left_right(x_k)
        x_q = tf.image.random_flip_left_right(x_q)
        # random grayscale
        grayscale_or_not = tf.random.uniform([2])
        if grayscale_or_not[0] > 0.5:
            x_k = tf.tile(tf.image.rgb_to_grayscale(x_k), (1, 1, 1, 3))
        if grayscale_or_not[1] > 0.5:
            x_q = tf.tile(tf.image.rgb_to_grayscale(x_q), (1, 1, 1, 3))
        return K.clip(x_k, 0, 255) / 255, K.clip(x_q, 0, 255) / 255
    
    def compute_output_shape(self, input_shape):
        ###### keras-fashion version ######
        # return (input_shape[0], self.queue_len+1)
        ###### gradient-tape version ###### 
        return (1)
    
    def compute_output_shape(self, input_shape):
        ###### keras-fashion version ######
        # return (input_shape[0], self.queue_len+1)
        ###### gradient-tape version ###### 
        return (1)

    def save_weights(self, epoch=0, loss=None):
        self.q_enc.save_weights(f"moco_weights_epoch_{epoch}_loss_{loss}.h5")
