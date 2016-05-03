import model as md
import random
from copy import deepcopy
import numpy as np
from _collections import defaultdict
from scipy.stats import kendalltau, spearmanr, pearsonr

#category = 'ranking'
#category = 'multi_labels'
category = 'binary_labels'

def getKendallDistance(true,predicted):
    
    kd = 0
    
    n = len(true)
    
    for i in range(0,n):
        for j in range(i+1,n):
            if not((true[i]<true[j] and predicted[i]<predicted[j])or(true[i]>true[j] and predicted[i]>predicted[j])):
                kd+=1
        
    
    kd = float(kd)/(n*(n+1)/2)
    
    return kd

def compute_rank_correlation_metrics(qids_to_labels):
            
    qids_to_ranks_true = qids_to_labels
    qids_to_ranks_predicted = defaultdict(list) 
    
        
    for qid in qids_to_ranks_true:
        n = len(qids_to_ranks_true[qid])
        qids_to_ranks_predicted[qid] = range(1,n+1)

    
    qids = qids_to_ranks_true.keys()
    
    total_kd_base = 0.0
    total_kd = 0.0
    
    for qid in qids:
        listA =  qids_to_ranks_true[qid]
        listB =  qids_to_ranks_predicted[qid]

        listC = deepcopy(listB)
        random.shuffle(listC)
        
        #kd, p_value = kendalltau(listA, listB)
        # kd_base,p_value = kendalltau(listA,listC)
        
        kd = getKendallDistance(listA, listB)
        kd_base = getKendallDistance(listA, listC)
        
        total_kd_base+=kd_base
        total_kd+=kd
        
        
    avg_kd_base = total_kd_base/len(qids)
    avg_kd = total_kd/len(qids)

    
    print 'average kendall tau correlation baseline = %f'%avg_kd_base
    print 'average kendall tau correlation = %f'%avg_kd
    
    return avg_kd_base,avg_kd


def precison_k(y_ranked,k):

    y_ranked_k = y_ranked[0:k]
    pos = float(y_ranked_k.count(1))
    pr = pos/k
        
    #print 'pr = %f'%pr
    return pr
        
#mean of precision at every relevant question     
def average_precision(y_ranked):
    
    avg_pr = 0.0
    rel = 0.0
    
    for r,y in enumerate(y_ranked):
        if y == 1:
            avg_pr+=precison_k(y_ranked, r+1) 
            rel+=1
            
    if rel!=0:
        avg_pr/=rel
    
    return avg_pr


def getBaselineRanking(qids_to_labels):
    
    qids_to_labels_baseline = defaultdict(list)
    
    #random shuffling to create a baseline ranking
    for qid in qids_to_labels:
        labels = deepcopy(qids_to_labels[qid])
        random.shuffle(labels)
        qids_to_labels_baseline[qid] = labels
    
    return qids_to_labels_baseline

def compute_MAP(qids_to_labels):
    
    n = len(qids_to_labels)
    MAP = 0.0
    
    for qid in qids_to_labels:
        y_ranked = qids_to_labels[qid]
        MAP+=average_precision(y_ranked)
        
    MAP/=n
    
    return MAP


def compute_map_metric(qids_to_labels):
    
    qids_to_labels_baseline = getBaselineRanking(qids_to_labels)
    
    #print qids_to_labels_baseline
    
    print 'MAP baseline ranking (random shuffle) = %f'%compute_MAP(qids_to_labels_baseline)
    print 'MAP with letor = %f'%compute_MAP(qids_to_labels)
    
def compute_precision20(qids_to_labels):
    
    pr20 = []
    
    for qid in qids_to_labels:
        y_ranked = qids_to_labels[qid]
        n = int(len(y_ranked)*0.2)
        y_top20 = y_ranked[0:n] 
        pr = float(y_top20.count(1))
        if n>0:
            pr20.append(pr/n)
        else:
            pr20.append(0)
      
    #print pr20
    
    return np.mean(pr20)

def compute_p20_metric(qids_to_labels):
    
    qids_to_labels_baseline = getBaselineRanking(qids_to_labels)
    
    print 'Precison@20 baseline (random shuffle) = %f'%compute_precision20(qids_to_labels_baseline)
    print 'Precison@20 with letor = %f'%compute_precision20(qids_to_labels)
    
    
def evaluate(output_file):

    context_question_list_ranked = md.readFile(output_file) 
    qids_to_labels = defaultdict(list)
    
    for item in context_question_list_ranked:
        item = list(item)
        qid = item[0]
        label = int(item[-1])
        qids_to_labels[qid].append(label)
        
    #labels are ranks 
    if category == 'ranking':
        compute_rank_correlation_metrics(qids_to_labels)
    #labels are 0/1 relevancy 
    elif category == 'binary_labels':
        compute_map_metric(qids_to_labels)
        compute_p20_metric(qids_to_labels)
    