import csv
from utils import configs as config
import numpy as np
import random
import pickle as pkl
import os
import math
import scipy.sparse as sp
import sys
from scipy.sparse import lil_matrix
import tensorflow as tf
import networkx as nx
from numpy.core.defchararray import lower


# build datasets from citeseer, cora and wiki
def build_sparse_nets_injection(dataset, train_ratio, inject_ratio=0.1, sign='content_inject'):
    """
        Args:
            dataset: name of dataset, i.e., wiki
            train_ratio: ratio of labeled nodes among all for training 
    """
    print("loading networks...")
    adjlist = {}
    nodenode_file = os.path.join(dataset, 'adjlist.txt')
    nodelabel_file = os.path.join(dataset, 'labels.txt')
    voca_file = os.path.join(dataset, 'vocabulary.txt')
    feature_file = os.path.join(dataset, 'content_token.txt')
    
    unique_edges = []
    with open(nodenode_file, encoding='utf-8') as nnfile, open(nodelabel_file, encoding="utf-8") as nlfile:
        for line in nnfile:
            params = line.replace('\n', '').split()
            root = int(params[0])
            neighbors = [int(v) for v in params[1:]]
            adjlist[root] = neighbors
            
            for ne in neighbors:
                s1 = str(root)+'_'+str(ne)
                s2 = str(ne)+'_'+str(root)
                if s1 not in unique_edges and s2 not in unique_edges:
                    unique_edges.append(s1)
            
        node_num = len(adjlist.keys())
        print("node_num",node_num)
        
        print("------------train index generation begin...-------")
        class_capacity = {}
        node_labels = {}
        label_per_class = {}  
        for line in nlfile:
            params = line.replace('\n', '').split()
            node = int(params[0])
            label = int(params[1].replace('L',''))
            
            if not label in label_per_class.keys():
                label_per_class[label] = [node]
            else:
                label_per_class[label].append(node)
            
            node_labels[node] = [label]
            
            if label not in class_capacity.keys():
                class_capacity[label] = 1
            else:
                class_capacity[label] += 1
        
        label_num = len(class_capacity.keys())
        print("label_num",label_num)
        print("actual samples per classes:",[(k, len(label_per_class[k])) for k in label_per_class.keys()])
                
        total_train_num = train_ratio
#         total_train_num = int(train_ratio * node_num)
        examples_per_label = int(total_train_num / len(class_capacity))
        label_capacity = {}
        for k in class_capacity.keys():
            if class_capacity[k] > examples_per_label:
                label_capacity[k] = examples_per_label
            else:
                label_capacity[k] = class_capacity[k]
        
        train_indexes = []
        cadidate_indexs = list(range(node_num))       
        print("examples_per_label", total_train_num, len(class_capacity), examples_per_label)
        while len(train_indexes)< total_train_num:
            nid = random.sample(cadidate_indexs, 1)[0]
#             print(nid, len(label_capacity))
            lbs = node_labels[nid]
            flag = False
            for lb in lbs:
                if lb in label_capacity.keys():
                    flag = True
                    break
            if flag:
                train_indexes.append(nid)
                cadidate_indexs.remove(nid)
                for lb in lbs:
                    if lb in label_capacity.keys():
                        label_capacity[lb] -= 1
                        if label_capacity[lb] <= 0:
                            label_capacity.pop(lb)
                    class_capacity[lb] -= 1
                    if class_capacity[lb] <= 0:
                        class_capacity.pop(lb)
                        
                    remain_num = total_train_num - len(train_indexes)
                    if len(label_capacity) <= 0:
                        examples_per_label = int(remain_num / len(class_capacity))
#                         print("examples_per_label",len(class_capacity), examples_per_label)
                        for k in class_capacity.keys():
                            if class_capacity[k] > examples_per_label:
                                label_capacity[k] = examples_per_label
                            else:
                                label_capacity[k] = class_capacity[k]
            
            if len(cadidate_indexs) <=0:
                break

        train_nodes = sorted([(idx, node_labels[idx]) for idx in train_indexes], key=lambda x: x)
        print('train_nodes:', train_nodes)
        train_label_per_class = {}     
        with open(os.path.join(dataset,str(train_ratio)+sign+str(inject_ratio)+"_train_index.txt"), 'w') as tiw:
            for chunk in train_nodes:
                if not chunk[1][0] in train_label_per_class.keys():
                    train_label_per_class[chunk[1][0]] = 1
                else:
                    train_label_per_class[chunk[1][0]] = train_label_per_class[chunk[1][0]] + 1
                ti_text = str(chunk[0]) + ' ' + str(chunk[1][0])
                tiw.write(ti_text+'\n') 
          
        print("training samples per classes:{}".format(train_label_per_class))
        print("------------train index generation end-------")
        tiw.close()
        
    nnfile.close()
    nlfile.close()
    
    adjlist = [(k,adjlist[k]) for k in sorted(adjlist.keys())]
    
    nnAdjM = lil_matrix((node_num, node_num))    
    for chunk in adjlist:
        root_node = chunk[0]
        adj_node = chunk[1]
        nnAdjM[root_node, adj_node] = 1    
        
    
    # remove structures
    if sign == 'remove_struct':
        remove_edges = random.sample(unique_edges,int(len(unique_edges)*inject_ratio))
        print('unique_edges:',len(unique_edges))
        print('remove_edges:',len(remove_edges))
        for s12 in remove_edges:
            s1 = int(s12.split('_')[0])
            s2 = int(s12.split('_')[1])
            nnAdjM[s1,s2]=0.0
            nnAdjM[s2,s1]=0.0
#     print(nnAdjM)
    
    # inject noise structures
    if sign == 'inject_struct':
        all_edges = []
        for i in range(node_num):
            for j in range(i+1, node_num):
                s1 = str(i)+'_'+str(j)
                if nnAdjM[i,j] == 0:
                    all_edges.append(s1)
        inject_edges = random.sample(all_edges, int(len(unique_edges)*inject_ratio))
        for s12 in inject_edges:
            s1 = int(s12.split('_')[0])
            s2 = int(s12.split('_')[1])
            nnAdjM[s1,s2]=1
            nnAdjM[s2,s1]=1
                
    
    nnAdjM = nnAdjM + np.identity(node_num)
    d_nnl_diag = np.squeeze(np.sum(np.array(nnAdjM), axis=1))
    d_nnl_inv_sqrt_diag = np.power(d_nnl_diag, -1/2)
    d_nnl_inv_sqrt = np.diag(d_nnl_inv_sqrt_diag)
    nnAdj_norm = np.dot(np.dot(d_nnl_inv_sqrt, nnAdjM), 
                          d_nnl_inv_sqrt)
                          
    nn_embeddings_identity = np.identity(n=node_num)
                          
    voca_dic = {}
    node_features = {}
    with open(voca_file, encoding='utf-8') as vofile, open(feature_file, encoding="utf-8") as fefile:
        for line in vofile:
            params = line.replace('\n', '').split('=')
            voca_dic[params[1]] = int(params[0])
        
        feature_num = len(voca_dic.keys())
        print('Total number of features:', feature_num)
        
        li = 0
        for line in fefile:
            params = line.replace('\n', '').split()
            word_list = []
            for word in params:
                word_list.append(voca_dic[word])
            node_features[li] = word_list
            li += 1
    nn_embeddings_feature = lil_matrix((node_num, feature_num))
    
    for i in range(node_num):
        feats = node_features[i]
        nn_embeddings_feature[i, feats] = 1
        
    # inject content noises
    if sign == 'content_inject':
        word_edges = []
        total_words = 0
        for i in range(node_num):
            for j in range(feature_num):
                s12 = str(i)+'_'+str(j)
                if nn_embeddings_feature[i, j] == 0:
                    word_edges.append(s12)
                else:
                    total_words += 1
        inject_words = random.sample(word_edges, int(total_words*inject_ratio))
        for s12 in inject_words:
            s1 = int(s12.split('_')[0])
            s2 = int(s12.split('_')[1])
            nn_embeddings_feature[s1,s2]=1
    
    print('nn_embeddings_feature_shape:',nn_embeddings_feature.shape)
        
    node_labels = [(k,node_labels[k]) for k in sorted(node_labels.keys())]    
    node_class = np.zeros([node_num,label_num])
    for chunk in node_labels:
        root_node = chunk[0]
        labels = chunk[1]
        node_class[root_node, labels] = 1
    
    csr_adj_nn = sp.csr_matrix(nnAdjM)
    csr_adj_nn_norm = sp.csr_matrix(nnAdj_norm)
    csr_node_class = sp.csr_matrix(node_class)
    csr_nn_embed_identity = sp.csr_matrix(nn_embeddings_identity)    
    csr_nn_embed_feature = sp.csr_matrix(nn_embeddings_feature)    
    
    f = open(os.path.join(dataset, str(train_ratio)+sign+str(inject_ratio)+'_m.adj'), 'wb')
    pkl.dump(csr_adj_nn, f)
    f.close()  
#     np.savetxt('out.nn_adj',csr_adj_nn.toarray()) 
    
    f = open(os.path.join(dataset, str(train_ratio)+sign+str(inject_ratio)+'_m_norm.adj'), 'wb')
    pkl.dump(csr_adj_nn_norm, f)
    f.close()   
    
    f = open(os.path.join(dataset, str(train_ratio)+sign+str(inject_ratio)+'_m.label'), 'wb')
    pkl.dump(csr_node_class, f)
    f.close()   
#     np.savetxt('out.label',csr_node_class.toarray())  
    
    f = open(os.path.join(dataset, str(train_ratio)+sign+str(inject_ratio)+'_identity.x'), 'wb')
    pkl.dump(csr_nn_embed_identity, f)
    f.close() 
           
    f = open(os.path.join(dataset, str(train_ratio)+sign+str(inject_ratio)+'_feature.x'), 'wb')
    pkl.dump(csr_nn_embed_feature, f)
    f.close()        
#     np.savetxt('out.feature',csr_nn_embed_feature.toarray()[0:20])  
    
# build_sparse_nets_injection(config.FILES.dblp, 80, inject_ratio=0.8, sign='content_inject')
# build_sparse_nets_injection(config.FILES.dblp, 80, inject_ratio=0.3, sign='remove_struct')
# build_sparse_nets_injection(config.FILES.dblp, 80, inject_ratio=0.8, sign='inject_struct')

# build datasets from citeseer, cora and wiki
def build_sparse_nets(dataset, train_ratio):
    """
        Args:
            dataset: name of dataset, i.e., wiki
            train_ratio: ratio of labeled nodes among all for training 
    """
    print("loading networks...")
    adjlist = {}
    nodenode_file = os.path.join(dataset, 'adjlist.txt')
    nodelabel_file = os.path.join(dataset, 'labels.txt')
    voca_file = os.path.join(dataset, 'vocabulary.txt')
    feature_file = os.path.join(dataset, 'content_token.txt')
    
    with open(nodenode_file, encoding='utf-8') as nnfile, open(nodelabel_file, encoding="utf-8") as nlfile:
        for line in nnfile:
            params = line.replace('\n', '').split()
            root = int(params[0])
            neighbors = [int(v) for v in params[1:]]
            adjlist[root] = neighbors
        node_num = len(adjlist.keys())
        print("node_num",node_num)
        
        print("------------train index generation begin...-------")
        class_capacity = {}
        node_labels = {}
        label_per_class = {}  
        for line in nlfile:
            params = line.replace('\n', '').split()
            node = int(params[0])
            label = int(params[1].replace('L',''))
            
            if not label in label_per_class.keys():
                label_per_class[label] = [node]
            else:
                label_per_class[label].append(node)
            
            node_labels[node] = [label]
            
            if label not in class_capacity.keys():
                class_capacity[label] = 1
            else:
                class_capacity[label] += 1
        
        label_num = len(class_capacity.keys())
        print("label_num",label_num)
        print("actual samples per classes:",[(k, len(label_per_class[k])) for k in label_per_class.keys()])
                
        total_train_num = train_ratio
#         total_train_num = int(train_ratio * node_num)
        examples_per_label = int(total_train_num / len(class_capacity))
        label_capacity = {}
        for k in class_capacity.keys():
            if class_capacity[k] > examples_per_label:
                label_capacity[k] = examples_per_label
            else:
                label_capacity[k] = class_capacity[k]
        
        train_indexes = []
        cadidate_indexs = list(range(node_num))       
        print("examples_per_label", total_train_num, len(class_capacity), examples_per_label)
        while len(train_indexes)< total_train_num:
            nid = random.sample(cadidate_indexs, 1)[0]
#             print(nid, len(label_capacity))
            lbs = node_labels[nid]
            flag = False
            for lb in lbs:
                if lb in label_capacity.keys():
                    flag = True
                    break
            if flag:
                train_indexes.append(nid)
                cadidate_indexs.remove(nid)
                for lb in lbs:
                    if lb in label_capacity.keys():
                        label_capacity[lb] -= 1
                        if label_capacity[lb] <= 0:
                            label_capacity.pop(lb)
                    class_capacity[lb] -= 1
                    if class_capacity[lb] <= 0:
                        class_capacity.pop(lb)
                        
                    remain_num = total_train_num - len(train_indexes)
                    if len(label_capacity) <= 0:
                        examples_per_label = int(remain_num / len(class_capacity))
#                         print("examples_per_label",len(class_capacity), examples_per_label)
                        for k in class_capacity.keys():
                            if class_capacity[k] > examples_per_label:
                                label_capacity[k] = examples_per_label
                            else:
                                label_capacity[k] = class_capacity[k]
            
            if len(cadidate_indexs) <=0:
                break

        train_nodes = sorted([(idx, node_labels[idx]) for idx in train_indexes], key=lambda x: x)
        print('train_nodes:', train_nodes)
        train_label_per_class = {}     
        with open(os.path.join(dataset,str(train_ratio)+"_train_index.txt"), 'w') as tiw:
            for chunk in train_nodes:
                if not chunk[1][0] in train_label_per_class.keys():
                    train_label_per_class[chunk[1][0]] = 1
                else:
                    train_label_per_class[chunk[1][0]] = train_label_per_class[chunk[1][0]] + 1
                ti_text = str(chunk[0]) + ' ' + str(chunk[1][0])
                tiw.write(ti_text+'\n') 
          
        print("training samples per classes:{}".format(train_label_per_class))
        print("------------train index generation end-------")
        tiw.close()
        
    nnfile.close()
    nlfile.close()
    
    adjlist = [(k,adjlist[k]) for k in sorted(adjlist.keys())]
    nnAdjM = lil_matrix((node_num, node_num))    
    for chunk in adjlist:
        root_node = chunk[0]
        adj_node = chunk[1]
        nnAdjM[root_node, adj_node] = 1    
    
    nnAdjM = nnAdjM + np.identity(node_num)
    d_nnl_diag = np.squeeze(np.sum(np.array(nnAdjM), axis=1))
    d_nnl_inv_sqrt_diag = np.power(d_nnl_diag, -1/2)
    d_nnl_inv_sqrt = np.diag(d_nnl_inv_sqrt_diag)
    nnAdj_norm = np.dot(np.dot(d_nnl_inv_sqrt, nnAdjM), 
                          d_nnl_inv_sqrt)
                          
    nn_embeddings_identity = np.identity(n=node_num)
                          
#     nn_embeddings_feature = []
#     with open(feature_file, encoding="utf-8") as fefile:
#         for line in fefile:
#             params = line.replace('\n', '').split()
#             node_id = params[0]
#             feats = [float(v) for v in params[1:]]
#             for i in range(len(feats)):
#                 if feats[i] != 0:
#                     feats[i] = 1.
#             nn_embeddings_feature.append(feats)
#     nn_embeddings_feature = np.array(nn_embeddings_feature)
    voca_dic = {}
    node_features = {}
    with open(voca_file, encoding='utf-8') as vofile, open(feature_file, encoding="utf-8") as fefile:
        for line in vofile:
            params = line.replace('\n', '').split('=')
            voca_dic[params[1]] = int(params[0])
        
        feature_num = len(voca_dic.keys())
        print('Total number of features:', feature_num)
        
        li = 0
        for line in fefile:
            params = line.replace('\n', '').split()
            word_list = []
            for word in params:
                word_list.append(voca_dic[word])
            node_features[li] = word_list
            li += 1
    nn_embeddings_feature = lil_matrix((node_num, feature_num))
    for i in range(node_num):
        feats = node_features[i]
        nn_embeddings_feature[i, feats] = 1
    
    print('nn_embeddings_feature_shape:',nn_embeddings_feature.shape)
        
    node_labels = [(k,node_labels[k]) for k in sorted(node_labels.keys())]    
    node_class = np.zeros([node_num,label_num])
    for chunk in node_labels:
        root_node = chunk[0]
        labels = chunk[1]
        node_class[root_node, labels] = 1
    
    csr_adj_nn = sp.csr_matrix(nnAdjM)
    csr_adj_nn_norm = sp.csr_matrix(nnAdj_norm)
    csr_node_class = sp.csr_matrix(node_class)
    csr_nn_embed_identity = sp.csr_matrix(nn_embeddings_identity)    
    csr_nn_embed_feature = sp.csr_matrix(nn_embeddings_feature)    
    
    f = open(os.path.join(dataset, str(train_ratio)+'_m.adj'), 'wb')
    pkl.dump(csr_adj_nn, f)
    f.close()  
#     np.savetxt('out.nn_adj',csr_adj_nn.toarray()) 
    
    f = open(os.path.join(dataset, str(train_ratio)+'_m_norm.adj'), 'wb')
    pkl.dump(csr_adj_nn_norm, f)
    f.close()   
    
    f = open(os.path.join(dataset, str(train_ratio)+'_m.label'), 'wb')
    pkl.dump(csr_node_class, f)
    f.close()   
#     np.savetxt('out.label',csr_node_class.toarray())  
    
    f = open(os.path.join(dataset, str(train_ratio)+'_identity.x'), 'wb')
    pkl.dump(csr_nn_embed_identity, f)
    f.close() 
           
    f = open(os.path.join(dataset, str(train_ratio)+'_feature.x'), 'wb')
    pkl.dump(csr_nn_embed_feature, f)
    f.close()        
#     np.savetxt('out.feature',csr_nn_embed_feature.toarray()[0:20])  
    
# build_sparse_nets(config.FILES.dblp, 80)


def load_data_injection(dataset, train_ratio, sign, inject_ratio, x_flag='feature'):
    """
    Loads input corpus from gcn/data directory

    m.x => the feature vectors of all nodes and labels as scipy.sparse.csr.csr_matrix object;
    m.adj => the adjacency matrix of node-node-label network as scipy.sparse.csr.csr_matrix object;
    m.label => the labels for all nodes as scipy.sparse.csr.csr_matrix objectt;
    train_index.txt => the indices of labeled nodes for supervised training as numpy.ndarray object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    train_ratio =str(train_ratio)
    names = [train_ratio+sign+str(inject_ratio)+"_"+x_flag+'.x', train_ratio+sign+str(inject_ratio)+'_m.adj', 
             train_ratio+sign+str(inject_ratio)+'_m_norm.adj', train_ratio+sign+str(inject_ratio)+'_m.label']
    
    objects = []
    for i in range(len(names)):
        with open(os.path.join(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, adj, adj_norm, label = tuple(objects)
    train_indexes = []
    label_counts = {}
    balance_num = 0
    with open(os.path.join(dataset,train_ratio+"_train_index.txt"), 'r') as tir:
        for line in tir:
            params = line.split()
            train_indexes.append(int(params[0]))
            if not params[1] in label_counts.keys():
                label_counts[params[1]] = [int(params[0])]
            else:
                label_counts[params[1]].append(int(params[0]))
    
            if balance_num < len(label_counts[params[1]]):
                balance_num = len(label_counts[params[1]])
    
    # print the class distribution
    label_dist = [(k,len(label_counts[k])) for k in sorted(label_counts.keys())]      
    print('label_distribution:', label_dist)
    print('balance_num:', balance_num)
#         id_text = tir.readline()
#         train_indexes = np.array([int(i) for i in id_text.split()])
    train_indexes = np.array(train_indexes)
    node_num = adj.shape[0]
    temp_indexes = np.setdiff1d(np.array(range(node_num)), train_indexes)
#     validation_indexes = np.array(random.sample(list(temp_indexes), int(0.1 * node_num)))
    test_indexes = np.array(random.sample(list(temp_indexes), 1000))
     
    test_idx = os.path.join(dataset, 'test_idx.txt')
    with open(test_idx, 'w') as tw:
        for id in test_indexes:
            tw.write(str(id) + ' ')
    test_idx = os.path.join(dataset, 'test_idx.txt')
    with open(test_idx, 'r') as tr:
        test_indexes = [int(id) for id in tr.readline().split()]
    
    test_indexes = np.array(test_indexes)
    print(x.shape, adj.shape, label.shape, train_indexes.shape, test_indexes.shape)
    
    return x, adj, adj_norm, label, train_indexes, test_indexes
def load_data(dataset, train_ratio, x_flag='feature'):
    """
    Loads input corpus from gcn/data directory

    m.x => the feature vectors of all nodes and labels as scipy.sparse.csr.csr_matrix object;
    m.adj => the adjacency matrix of node-node-label network as scipy.sparse.csr.csr_matrix object;
    m.label => the labels for all nodes as scipy.sparse.csr.csr_matrix objectt;
    train_index.txt => the indices of labeled nodes for supervised training as numpy.ndarray object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    train_ratio =str(train_ratio)
    names = [train_ratio+"_"+x_flag+'.x', train_ratio+'_m.adj', 
             train_ratio+'_m_norm.adj', train_ratio+'_m.label']
    
    objects = []
    for i in range(len(names)):
        with open(os.path.join(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, adj, adj_norm, label = tuple(objects)
    train_indexes = []
    label_counts = {}
    balance_num = 0
    with open(os.path.join(dataset,train_ratio+"_train_index.txt"), 'r') as tir:
        for line in tir:
            params = line.split()
            train_indexes.append(int(params[0]))
            if not params[1] in label_counts.keys():
                label_counts[params[1]] = [int(params[0])]
            else:
                label_counts[params[1]].append(int(params[0]))
    
            if balance_num < len(label_counts[params[1]]):
                balance_num = len(label_counts[params[1]])
    
    # print the class distribution
    label_dist = [(k,len(label_counts[k])) for k in sorted(label_counts.keys())]      
    print('label_distribution:', label_dist)
    print('balance_num:', balance_num)
#         id_text = tir.readline()
#         train_indexes = np.array([int(i) for i in id_text.split()])
    train_indexes = np.array(train_indexes)
    node_num = adj.shape[0]
    temp_indexes = np.setdiff1d(np.array(range(node_num)), train_indexes)
#     validation_indexes = np.array(random.sample(list(temp_indexes), int(0.1 * node_num)))
    test_indexes = np.array(random.sample(list(temp_indexes), 1000))
     
#     test_idx = os.path.join(dataset, 'test_idx_temp.txt')
#     with open(test_idx, 'w') as tw:
#         for id in test_indexes:
#             tw.write(str(id) + ' ')
    test_idx = os.path.join(dataset, 'test_idx.txt')
    with open(test_idx, 'r') as tr:
        test_indexes = [int(id) for id in tr.readline().split()]
    
    test_indexes = np.array(test_indexes)
    print(x.shape, adj.shape, label.shape, train_indexes.shape, test_indexes.shape)
    
    return x, adj, adj_norm, label, train_indexes, test_indexes



def load_data_public(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print(x.shape, y.shape, tx.shape)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return features, adj, labels, idx_train, idx_test

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def build_imageset(dataset, dir):
    
    root = 'F:/myeclipse/flikr_dataset/dataset'
    parse_file = os.path.join('F:/myeclipse/flikr_dataset/flickrXml/data', dataset+".txt")
    flickr_edges = 'F:/myeclipse/flikr_dataset/flickrEdges.txt'
    
    img_idmap = {}
    label_idmap = {}   
    node_labelmap = {} 
    with open(parse_file, encoding='utf-8') as pfr:
        for line in pfr:
            param1 = line.replace('\n','').split('%%')
            param2 = param1[1].split('==')[1].split('@@@###')
            param3 = param2[1].split('###')
            
            img_id = param1[0]
            node_id = param2[0]
            img_idmap[node_id] = img_id
            
            node_tags = []
            for tag in param3:
                tag = tag.lower()
#                 if not tag in label_idmap.keys():
#                     label_idmap[tag] = 'L' + str(len(label_idmap))
                node_tags.append(tag)
            node_labelmap[node_id] = node_tags
    
    print('node num:{}'.format(len(node_labelmap)))
    
    node_labels = {}
    node_idmap = {}  
    node_adj = {}  
    nodeid_imgid = {}
    with open(flickr_edges, 'r') as fer:
        for _ in range(4):
            next(fer)
        for line in fer:
            params = line.split()
            snode = params[0]
            enode = params[1]
            
            flag = False
            if snode in node_labelmap.keys():
                if not snode in node_idmap.keys():
                    node_idmap[snode] = len(node_idmap)
                if not node_idmap[snode] in node_adj.keys():
                    node_adj[node_idmap[snode]] = []
                    nodeid_imgid[node_idmap[snode]] = img_idmap[snode]
                    
                    node_tags = []
                    for tag in node_labelmap[snode]:
                        if not tag in label_idmap.keys():
                            label_idmap[tag] = 'L' + str(len(label_idmap))
                        node_tags.append(label_idmap[tag])
                    node_labels[node_idmap[snode]] = node_tags
                    
                flag = True
                   
            if enode in node_labelmap.keys():
                if not enode in node_idmap.keys():
                    node_idmap[enode] = len(node_idmap)
                if not node_idmap[enode] in node_adj.keys():
                    node_adj[node_idmap[enode]] = []   
                    nodeid_imgid[node_idmap[enode]] = img_idmap[enode] 
                    
                    node_tags = []
                    for tag in node_labelmap[enode]:
                        if not tag in label_idmap.keys():
                            label_idmap[tag] = 'L' + str(len(label_idmap))
                        node_tags.append(label_idmap[tag])
                    node_labels[node_idmap[enode]] = node_tags
                    
                if flag and node_idmap[snode] not in node_adj[node_idmap[enode]]:
                    node_adj[node_idmap[enode]].append(node_idmap[snode])   
                    node_adj[node_idmap[snode]].append(node_idmap[enode])   
    
    print(label_idmap)
    label_list = os.path.join(root, dir+'/labels.txt')
    adj_list = os.path.join(root, dir+'/adjlist.txt')
     
    with open(label_list, 'w') as llw:
        for k in node_labels.keys():
            lbs = ' '.join([lb for lb in node_labels[k]])
            llw.write(str(k) + ' ' + lbs + '\n')
             
    with open(adj_list, 'w') as alw:
        for k in node_adj.keys():
            line = str(k) + ' ' + ' '.join([str(i) for i in node_adj[k]])
            alw.write(line + '\n')
             
    import shutil
    for k in nodeid_imgid.keys():
        source = os.path.join('F:/myeclipse/flikr_dataset/flickrXml/mirflickr',
                             'im'+nodeid_imgid[k]+'.jpg')
        destination = os.path.join(root, dir+'/feat/'+str(k)+'.jpg')
        shutil.copyfile(source, destination) 
        
                    
    print('node num: {}, label num:{}'.format(len(node_adj), len(label_idmap)))              
    
# build_imageset('photosCLEF', 'ImageCLEF')
# build_imageset('photosMIR', 'MIR')
