# -*- coding: utf-8 -*-

import tensorflow as tf

# define the paths of the datasets
# citeseer 
tf.flags.DEFINE_string("citeseer", "../datasets/citeseer", "")
# cora 
tf.flags.DEFINE_string("cora", "../datasets/cora", "")
# dblp 
tf.flags.DEFINE_string("dblp", "../datasets/dblp", "")
# pubmed 
tf.flags.DEFINE_string("pubmed", "../datasets/pubmed", "")
# MIR 
tf.flags.DEFINE_string("MIR", "../datasets/MIR", "")
# ImageCLEF 
tf.flags.DEFINE_string("ImageCLEF", "../datasets/ImageCLEF", "")


tf.flags.DEFINE_integer("hidden_dim", 100, "Dimensionality of hidden networks in GAN")
tf.flags.DEFINE_integer("d", 30, "Dimensionality of content network construction")
tf.flags.DEFINE_integer("h", 30, "Dimensionality of GCN hidden layer")

# tf.flags.DEFINE_float('emb_dropout_prob', 0.2, 'Dropout probability of embedding layer')
tf.flags.DEFINE_float('dropout_prob', 0.5, 'Dropout probability of output layer')

tf.flags.DEFINE_float('learning_rate', 2e-3, 'Initial learning rate.')
# tf.flags.DEFINE_float('learning_rate_gan', 5e-4, 'Initial learning rate.')
# tf.flags.DEFINE_float('weight_decay', 3e-3, 'Weight for L2 loss on embedding matri')
tf.flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matri')
tf.flags.DEFINE_float('early_stopping', 10, 'allow elary stop')

# FILES = tf.flags.FLAGS
FILES = tf.flags.FLAGS