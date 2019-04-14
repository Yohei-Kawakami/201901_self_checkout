from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


class Arcfacelayer(Layer):
    def __init__(self, output_dim, s=30, m=0.50, easy_magin=False):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.easy_magin = easy_magin
        super(Arcfacelayer, self).__init__()
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Arcfacelayer, self).build(input_shape)
        
    def call(self, x):
        y = x[1]
        x_normalize = tf.math.l2_normalize(x[0]) #|x = x'/ ||x'||2
        k_normalize = tf.math.l2_normalize(self.kernel) # Wj = Wj' / ||Wj'||2
        
        cos_m = K.cos(self.m)
        sin_m = K.sin(self.m)
        th = K.cos(np.pi - self.m)
        mm = K.sin(np.pi - self.m) * self.m
        
        cosine = K.dot(x_normalize, k_normalize) # W.Txの内積
        
        sine = K.sqrt(1.0 - K.square(cosine))
        phi = cosine * cos_m - sine * sin_m
        
        if self.easy_magin:
            phi = tf.where(cosine > 0, phi, cosine) 
            
        else:
            phi = tf.where(cosine > th, phi, cosine - mm) 
        
        output = (y * phi) + ((1.0 - y) * cosine) # true cos(θ+m), False cos(θ)
        output *= self.s
        
        return output
    
    def compute_output_shape(self, input_shape):
        
        return (input_shape[0][0], self.output_dim)
        #return self.output_dim

