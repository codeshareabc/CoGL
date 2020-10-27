import tensorflow as tf
import numpy as np
import matplotlib.gridspec as gridspec


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
    
class Discriminator:
    def __init__(self, x_dim, h_dim):
        
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.var={}
        
        self.var['D_W1'] = glorot([x_dim, h_dim], name='D_W1')
        self.var['D_b1'] = zeros([h_dim], name='D_b1')
        
        self.var['D_W2'] = glorot([h_dim, 1], name='D_W2')
        self.var['D_b2'] = zeros([1], name='D_b2')
       
    def call(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.var['D_W1']) + self.var['D_b1'])
        D_logit = tf.matmul(D_h1, self.var['D_W2']) + self.var['D_b2']
        D_prob = tf.nn.sigmoid(D_logit)
        
        return D_prob, D_logit
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs) 
      
       
       
       
       
       
       
       
       
       
       
       
       
       
        