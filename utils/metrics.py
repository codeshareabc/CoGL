import numpy as np

def pre_calculate(category_nodes, category_nodes_pred):
        pred_label_counts = []
        true_label_counts = []
        
        true_positive = []
        false_positive = []
        false_negative = []
        
        for label in category_nodes_pred.keys():
            pred_nodes = category_nodes_pred[label]
            true_nodes = []
            if label in category_nodes.keys():
                true_nodes = category_nodes[label]
            
            pred_label_counts.append(len(pred_nodes))
            true_label_counts.append(len(true_nodes))
            
            hit_num = 0
            for pnode in pred_nodes:
                if pnode in true_nodes:
                    hit_num += 1
            true_positive.append(hit_num) 
            false_positive.append(len(pred_nodes) - hit_num)
            
            false_negative.append(len(true_nodes) - hit_num)
        
        return true_positive, false_positive, false_negative

def evaluate(pred_true_labels):
#     print(len(pred_true_labels))
    pred_category = {}
    true_category = {}
    
    for i in pred_true_labels.keys():
        pred_label_i = pred_true_labels[i][0]
#         print(len(pred_label_i))
        true_label_i = pred_true_labels[i][1]
        
        for c1 in pred_label_i:
            if c1 in pred_category:
                pred_category[c1].append(i)
            else:
                pred_category[c1] = [i]
        for c2 in true_label_i:
            if c2 in true_category:
                true_category[c2].append(i)
            else:
                true_category[c2] = [i]
    num_classes = len(pred_category)            
    true_positive, false_positive, false_negative= pre_calculate(true_category, pred_category) 
    
    micro_recall = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_negative))
    micro_precision = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
    micro_f1 = (2 * np.sum(true_positive)) / (2*np.sum(true_positive) + np.sum(false_positive) + np.sum(false_negative))
    # Calculate the macro-F1
#     macro_recall = .0
    macro_precision = .0
    macro_f1=.0
    for i_ in range(num_classes):
#         macro_recall += (true_positive[i_] / (true_positive[i_]+false_negative[i_]))
        macro_precision += (true_positive[i_] / (true_positive[i_]+false_positive[i_]))
        macro_f1 += (2 * true_positive[i_]) / (2 * true_positive[i_] + false_negative[i_]+false_positive[i_])
#     macro_recall /= num_classes
    macro_precision /= num_classes
    macro_f1 /= num_classes
    
    return micro_recall, micro_f1, macro_f1

def evaluate_bycategory(pred_true_labels):
#     print(len(pred_true_labels))
    pred_category = {}
    true_category = {}
    
    for i in pred_true_labels.keys():
        pred_label_i = pred_true_labels[i][0]
#         print(len(pred_label_i))
        true_label_i = pred_true_labels[i][1]
        
        for c1 in pred_label_i:
            if c1 in pred_category:
                pred_category[c1].append(i)
            else:
                pred_category[c1] = [i]
        for c2 in true_label_i:
            if c2 in true_category:
                true_category[c2].append(i)
            else:
                true_category[c2] = [i]
    num_classes = len(pred_category)            
    true_positive, false_positive, false_negative= pre_calculate(true_category, pred_category) 
    
#     micro_recall = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_negative))
#     micro_precision = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
#     micro_f1 = (2 * np.sum(true_positive)) / (2*np.sum(true_positive) + np.sum(false_positive) + np.sum(false_negative))
    # Calculate the macro-F1
#     macro_recall = .0
    macro_f1s = {}
    for i_ in range(num_classes):
        macro_f1s[pred_category.popitem()[0]] = (2 * true_positive[i_]) / (2 * true_positive[i_] + false_negative[i_]+false_positive[i_])
#     macro_recall /= num_classes
    return macro_f1s
    
    
    