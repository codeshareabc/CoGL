import numpy as np
import tensorflow as tf
from utils import sparse
import scipy.sparse


conv1d = tf.layers.conv1d

class BuildGraphStruct:
    
    def __init__(self, out_size, holders, name, act=tf.nn.relu, 
                  feat_dropout=True, biase=True):
        
        self.out_size = out_size
        self.dropout_prob = tf.dtypes.cast(holders['dropout_prob'], tf.float64)
        self.name = name
        self.act = act
        self.feat_dropout = feat_dropout
        self.biase = biase
        
    
    def call(self, X):  
        X = tf.dtypes.cast(X, tf.float64) 
        if self.feat_dropout:
            X = tf.nn.dropout(X, 1.0-self.dropout_prob)
        u_omega = tf.get_variable(self.name, [X.shape[0]], dtype=tf.float64, 
                                  initializer=tf.keras.initializers.glorot_normal())
        
        # project the input feature dimension to fit the output size
        feats = X[tf.newaxis]
        proj_X = tf.layers.conv1d(feats, self.out_size, 1, use_bias=False)
#         
#         # single feedforward neural network to learn edge weights
#         f_1 = tf.layers.conv1d(feats, 1, 1)
#         f_2 = tf.layers.conv1d(feats, 1, 1)
        f_1 = tf.reduce_sum(proj_X, 2, keep_dims=True)
        f_2 = tf.reduce_sum(proj_X, 2, keep_dims=True)
        logits = tf.abs(f_1 - tf.transpose(f_2, [0, 2, 1]))
#         
        logits = tf.squeeze(logits)

#         logits = tf.nn.softmax(tf.matmul(X, tf.transpose(X)))
        
        edge_w = tf.nn.softmax(tf.nn.leaky_relu(tf.multiply(u_omega, logits)))
        
        edge_w = tf.dtypes.cast(edge_w, tf.float32) 
        edge_w = tf.contrib.layers.dense_to_sparse(edge_w)
        
        self.vars = tf.trainable_variables()
        
        return edge_w
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)       
    
    