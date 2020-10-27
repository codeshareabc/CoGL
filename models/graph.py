"""Spectral Graph Convolutional filter cell."""
import numpy as np
import tensorflow as tf
import os

def _dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class GraphConvLayer:
    def __init__(self, input_dim, output_dim, name, holders, act=tf.nn.relu,
                  dropout=False, bias=True):
#                  name, act=tf.nn.relu, bias=False, dropout=):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act=act
        self.bias = bias
        self.dropout = dropout
        self.var={}
        
        with tf.variable_scope(name):
            
            with tf.name_scope('weights'):
                self.var['w']=glorot([input_dim, output_dim],
                                          name='w')
#                 self.var['w_c']=glorot([input_dim, output_dim],
#                                           name='w_')
            if self.bias:
                self.var['b']=zeros([output_dim],
                                          name='b')
#                 self.var['b_c']=zeros([output_dim],
#                                           name='b_')
            if self.dropout:
                self.dropout_prob = holders['dropout_prob']
            else:
                self.dropout_prob = 0.
            self.num_features_nonzero = holders['num_features_nonzero']
                    
    def call(self, adj_norm_c, x, x_c, sparse=False):  
        
        adj_norm = adj_norm_c[0]
        edge_w = adj_norm_c[1]
        
        if sparse:
            x = sparse_dropout(x, 1-self.dropout_prob, self.num_features_nonzero)
            x_c = sparse_dropout(x_c, 1-self.dropout_prob, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout_prob)
            x_c = tf.nn.dropout(x_c, 1-self.dropout_prob)
        hw = _dot(x=x, y=self.var['w'], sparse=sparse)
        hw_c = _dot(x=x_c, y=self.var['w'], sparse=sparse)
          
        ahw = _dot(x=adj_norm, y=hw, sparse=True)
        ahw_c = _dot(x=edge_w, y=hw_c, sparse=True)
        
        if self.bias:
            embed_out = self.act(tf.add(ahw, self.var['b']))   
            embed_out_c = self.act(tf.add(ahw_c, self.var['b'])) 
        else:
            embed_out = self.act(ahw)
            embed_out_c = self.act(ahw_c)
              
        return embed_out, embed_out_c  
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)        