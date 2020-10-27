import os
import time
import tensorflow as tf
import numpy as np
import models.graph_gcn as mg
import scipy.sparse
from utils import data_process, sparse
from utils import configs, metrics

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

train_ratio = 120
dataset = configs.FILES.citeseer
lrate_gcn = configs.FILES.learning_rate
print("train_ratio:",train_ratio)

x, _, adj_norm, labels, train_indexes, test_indexes = data_process.load_data(dataset, str(train_ratio), 
                                                                                x_flag='feature')

node_num = adj_norm.shape[0]
label_num = labels.shape[1]

print("node_num:",node_num)
print("train_num:",len(train_indexes))
print("train_num:",len(test_indexes))

adj_norm_tuple = sparse.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))
feat_x_nn_tuple = sparse.sparse_to_tuple(scipy.sparse.coo_matrix(x))

print(feat_x_nn_tuple)

# node-node network train and validate masks
nn_train_mask = np.zeros([node_num,])
nn_test_mask = np.zeros([node_num,])

for i in train_indexes:
    nn_train_mask[i] = 1
    nn_test_mask[i] = 0
    
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
                'dropout_prob': tf.placeholder_with_default(0., shape=()),
                'num_features_nonzero': tf.placeholder(tf.int32)
                }


# the first layer
# feat_embeds, proj_X = feat2struct.BuildGraphStruct(out_size = 10,
#                                         dropout_prob = 0.3,
#                                         act = tf.nn.relu,
#                                         feat_dropout = True)(X = x.toarray())

t_model = mg.GraphConvLayer(input_dim=x.shape[-1],
                           output_dim=configs.FILES.h,
                           name='nn_fc1',
                           holders=placeholders,
                           act=tf.nn.relu,
                           dropout=True)

nn_fc1 = t_model(adj_norm=ph['adj_norm'],
                           x=ph['x'], sparse=True)
                            

# the second layer
nn_dl = mg.GraphConvLayer(input_dim=configs.FILES.h,
                           output_dim=label_num,
                           name='nn_dl',
                           holders=placeholders,
                           act=tf.nn.softmax,
                           dropout=True)(adj_norm=ph['adj_norm'],
                                           x=nn_fc1)  


def masked_sigmoid_softmax_cross_entropy(preds, labels, mask):
    """Sigmoid softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    for var in t_model.var.values():
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
    
    loss = masked_sigmoid_softmax_cross_entropy(preds=nn_dl, 
                                                labels=ph['labels'], mask=ph['mask'])
        
    accuracy, correct_prediction = masked_accuracy(preds=nn_dl, 
                               labels=ph['labels'], mask=ph['mask'])
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate_gcn)
    opt_op = optimizer.minimize(loss)    

feed_dict_train = {ph['adj_norm']: adj_norm_tuple,
                      ph['x']: feat_x_nn_tuple,
                    ph['labels']: labels.toarray(),
                      ph['mask']: nn_train_mask,
                      placeholders['dropout_prob']: configs.FILES.dropout_prob,
                      placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape,
                      }
feed_dict_val = {ph['adj_norm']: adj_norm_tuple,
                    ph['x']: feat_x_nn_tuple,
                    ph['labels']: labels.toarray(),
                    ph['mask']: nn_test_mask,
                    placeholders['dropout_prob']: 0.,
                    placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape
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
    _, train_loss, train_acc, train_nn_dl = sess.run((opt_op, loss, accuracy, nn_dl), 
                                                              feed_dict=feed_dict_train)
    
    if epoch % save_every == 0:
        val_acc, test_nn_dl = sess.run((accuracy, nn_dl), feed_dict=feed_dict_val)
        
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc),
              "test_acc=", "{:.5f}".format(val_acc),
              "time=", "{:.5f}".format(time.time() - t))
    
    if epoch % 20 == 0:
        times.append(time.time() - t)



