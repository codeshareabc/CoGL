import os
import time
import tensorflow as tf
import numpy as np
import models.graph as mg
import models.adversarialNets as ma
import scipy.sparse
from utils import data_process, sparse
from utils import configs, metrics, feat2struct

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

train_ratio = 120
batch_size = 800
beta = 0.8
namda = 0.4
print('batch_size:',batch_size)
dataset = configs.FILES.citeseer
lrate_gcn = configs.FILES.learning_rate
print("train_ratio:",train_ratio)

x, _, adj_norm, labels, train_indexes, test_indexes = data_process.load_data(dataset, str(train_ratio), 
                                                                                x_flag='feature')


node_num = adj_norm.shape[0]
label_num = labels.shape[1]

adj_norm_tuple = sparse.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))
feat_x_nn_tuple = sparse.sparse_to_tuple(scipy.sparse.coo_matrix(x))

# node-node network train and validate masks
nn_train_mask = np.zeros([node_num,])
nn_test_mask = np.zeros([node_num,])

# batch training indexes for gan
gan_idx = tf.placeholder(tf.int32, shape=(batch_size,))
real_sample_x = tf.placeholder(tf.float32, shape=[None, label_num])

for i in train_indexes:
    nn_train_mask[i] = 1
#     nn_test_mask[i] = 0
    
for i in test_indexes:
    nn_test_mask[i] = 1
    
# TensorFlow placeholders
ph = {
      'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_norm"),
      'x': tf.sparse_placeholder(tf.float32, name="features"),
      'labels': tf.placeholder(tf.float32, name="node_labels"),
      'mask': tf.placeholder(tf.int32, shape=(node_num,))
      }

placeholders = {
                'dropout_prob': tf.placeholder(tf.float32),
                'num_features_nonzero': tf.placeholder(tf.int32)
                }


# the first layer
edge_model = feat2struct.BuildGraphStruct(out_size = configs.FILES.d,
                                        holders=placeholders,
                                        name='e1',
                                        act = tf.nn.relu,
                                        feat_dropout = True)
edge_w = edge_model(X = x.toarray())

t_model = mg.GraphConvLayer(input_dim=x.shape[-1],
                           output_dim=configs.FILES.h,
                           name='nn_fc1',
                           holders=placeholders,
                           act=tf.nn.relu,
                           dropout=True)
  
nn_fc1, embeds_c= t_model(adj_norm_c=[ph['adj_norm'],edge_w],
                           x=ph['x'], x_c=ph['x'], sparse=True)


# the second layer
nn_dl, embeds_c2= mg.GraphConvLayer(input_dim=configs.FILES.h,
                           output_dim=label_num,
                           name='nn_dl',
                           holders=placeholders,
                           act=tf.nn.softmax,
                           dropout=True)(adj_norm_c=[ph['adj_norm'],edge_w],
                                           x=nn_fc1, x_c=embeds_c)
                           

# the discriminative network
real_x = tf.gather(embeds_c2, gan_idx)
fake_x = tf.gather(nn_dl, gan_idx)

gan_model = ma.Discriminator(x_dim=label_num, h_dim=configs.FILES.hidden_dim)
D_real, D_logit_real = gan_model(real_sample_x)
D_fake, D_logit_fake = gan_model(fake_x)


def frobenius_distance(embeds_c2, original_x, edge_w, original_adj):
    original_x=tf.nn.softmax(original_x)
    embeds_c2=tf.nn.softmax(embeds_c2)
    loss =tf.norm(tf.matmul(original_x,tf.transpose(original_x))-tf.matmul(embeds_c2,tf.transpose(embeds_c2)),
                       ord='fro', axis=(0,1))
#     loss += tf.norm(edge_w-original_adj, ord='fro', axis=(0,1))
#     loss += tf.norm(edge_w, ord='fro', axis=(0,1))
    
#     for var in edge_model.vars:
#         var = tf.cast(var, dtype=tf.float32)
#         loss += configs.FILES.weight_decay * tf.nn.l2_loss(var)
    
    return loss

def gan_sigmoid_cross_entropy(D_real_preds, D_fake_preds):
    
    D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_preds, 
                                                          labels=tf.ones_like(D_real_preds))
    D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_preds, 
                                                          labels=tf.zeros_like(D_fake_preds))
    loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_preds, 
                                                     labels=tf.ones_like(D_fake_preds))
    
    loss_d = tf.reduce_mean(D_loss_real) + tf.reduce_mean(D_loss_fake)
    loss_g = tf.reduce_mean(loss_g)
#     return tf.reduce_mean(D_loss_real) + tf.reduce_mean(D_loss_fake) + tf.reduce_mean(loss_g)
    return loss_d, loss_g
    
    

def masked_sigmoid_softmax_cross_entropy(preds, labels, mask):
    """Sigmoid softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
#     for var in tf.trainable_variables():
    for var in t_model.var.values():
        var = tf.cast(var, dtype=tf.float32)
        loss += configs.FILES.weight_decay * tf.nn.l2_loss(var)
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    
    return tf.reduce_mean(accuracy_all), correct_prediction

# calculate the classification accuracy per classes   
def precision_per_class(preds, labels, mask):
    import heapq
    mask = mask.astype(int)
    labels = labels.astype(int)
    val_indexes = np.where(mask==1)[0]
    pred_true_labels = {}
    
    y_true = []
    y_pred = []    
    
    for i in val_indexes:
        pred_probs_i = preds[i]
        true_raw_i = labels[i]
        
        pred_label_i = heapq.nlargest(np.sum(true_raw_i),range(len(pred_probs_i)), 
                                      pred_probs_i.take)
        true_label_i = np.where(true_raw_i==1)[0]
        pred_true_labels[i] = (pred_label_i, true_label_i)
        
        y_true.append(true_label_i)
        y_pred.append(pred_label_i)
        
    accuracy_per_classes = metrics.evaluate(pred_true_labels)
    
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    
    mat = confusion_matrix(y_true, y_pred)
    print(mat)        
        
    fpr = dict()
    tpr = dict()
    test_y = labels[val_indexes]
    test_pred = preds[val_indexes]
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), test_pred.ravel())
    auc = auc(fpr["micro"], tpr["micro"])
    
    return accuracy_per_classes, auc
    
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    test_y = labels[val_indexes]
    test_pred = preds[val_indexes]
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), test_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('micro_auc=',roc_auc["micro"])
    
    return accuracy_per_classes

with tf.name_scope('optimizer'):
    # graph construct loss L_{cont}
    graph_build_loss = frobenius_distance(embeds_c2,
                                          tf.dtypes.cast(x.toarray(), tf.float32),
                                          tf.sparse.to_dense(edge_w),
                                          adj_norm.toarray())
    
    # discriminator training loss L_{gan}
    d_loss, g_loss = gan_sigmoid_cross_entropy(D_real_preds=D_logit_real, 
                                               D_fake_preds=D_logit_fake)
    
    # semi-supervised node classification loss L_{gcn}
    class_loss = masked_sigmoid_softmax_cross_entropy(preds=nn_dl, 
                                                labels=ph['labels'], 
                                                mask=ph['mask'])
#     
#     loss = 0.2*graph_build_loss + d_loss + class_loss
    loss1 = namda*graph_build_loss + beta*d_loss + class_loss
    loss2 = namda*graph_build_loss + beta*g_loss + class_loss
        
    accuracy, correct_prediction = masked_accuracy(preds=nn_dl, 
                               labels=ph['labels'], mask=ph['mask'])
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate_gcn)
#     opt_op = optimizer.minimize(loss)    
    opt_op1 = optimizer.minimize(loss1)    
    opt_op2 = optimizer.minimize(loss2)    

feed_dict_train = {ph['adj_norm']: adj_norm_tuple,
                      ph['x']: feat_x_nn_tuple,
                    ph['labels']: labels.toarray(),
                      ph['mask']: nn_train_mask,
                      placeholders['dropout_prob']: configs.FILES.dropout_prob,
                      placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape,
                      gan_idx: None,
                      real_sample_x: None
                      }
feed_dict_val = {ph['adj_norm']: adj_norm_tuple,
                    ph['x']: feat_x_nn_tuple,
                    ph['labels']: labels.toarray(),
                    ph['mask']: nn_test_mask,
                    placeholders['dropout_prob']: 0.,
                    placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape,
                    gan_idx: None
                    }

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 400
save_every = 1    
    
t = time.time()
# Train model
times = []
for epoch in range(epochs):
    # Training node embedding
#     _, train_loss = sess.run(
#         (opt_op, loss), feed_dict=feed_dict_train)
    # obtain the training sample indexes for gan
    begin = epoch * batch_size
    end = epoch * batch_size + batch_size
    batch_idx = []
    for i in range(begin, end):
        idx = i % node_num
        batch_idx.append(idx)

    if epoch % save_every == 0:
        feed_dict_val.update(({gan_idx: batch_idx}))
        val_acc, test_nn_dl, real_x_v = sess.run((accuracy, nn_dl, real_x), feed_dict=feed_dict_val)
    feed_dict_train.update(({gan_idx: batch_idx, real_sample_x: real_x_v}))
    sess.run((opt_op1, graph_build_loss, accuracy, nn_dl),feed_dict=feed_dict_train) 
    _, dd_loss, gg_loss, cclass_loss, train_acc, train_nn_dl = sess.run((opt_op2, d_loss, g_loss, class_loss, accuracy, nn_dl), 
                                                              feed_dict=feed_dict_train) 
    
    ggraph_build_lossa.append(str("{:.5f}".format(ggraph_build_loss))) 
    dd_lossa.append(str("{:.5f}".format(dd_loss))) 
    gg_lossa.append(str("{:.5f}".format(gg_loss))) 
    cclass_lossa.append(str("{:.5f}".format(cclass_loss))) 

    print("Epoch:", '%04d' % (epoch + 1),
          "dd_loss=", "{:.5f}".format(dd_loss),
          "gg_loss=", "{:.5f}".format(gg_loss),
          "cclass_loss=", "{:.5f}".format(cclass_loss),
          "train_acc=", "{:.5f}".format(train_acc),
          "test_acc=", "{:.5f}".format(val_acc),
          "time=", "{:.5f}".format(time.time() - t))

    if epoch % 20 == 0:
        times.append(time.time() - t)
        
