from _collections import defaultdict
import featureExtraction as FE
from copy import deepcopy
import random
import os
import sys
from scipy.stats import kendalltau, spearmanr, pearsonr

allFeatures = []

def readFile(f,file_type):
    
    text_question_list = []
    
    with open(f,"r") as fle:
        for line in fle:
            line = line.rstrip('\n')
            
            if line=="":continue
            
            parts = line.split('\t')

            qid = parts[0]
            
            sentence = parts[1]
            question = parts[2]
            
            if file_type!='output':score = parts[3]
            #else:score = parts[4]
            
            rank = parts[-1]
            
            if file_type=='train':
                inp = [qid,sentence,question,score,rank]
            else:
                inp = [qid,sentence,question,rank]
            
            text_question_list.append(tuple(inp))
    
    #print question_answer_list

    return text_question_list

'''

def normalizeFeatures(featureVectors):
    
    allFeatures = set()
    
    for vector in featureVectors:
        allFeatures.update(vector.features.keys())
    
    #all the features in any feature vector
    print list(allFeatures)
    
'''

#there can be new features in test data
#test set features should be extracted only based on training set features
#fix this
#this needs to be handled
def getAllFeatures(questionList):
    
    allFeaturesSet = set()
    
    for question in questionList:
        allFeaturesSet.update(question.features.keys())
    
    #all the features in any feature vector
    global allFeatures
    allFeatures = deepcopy(list(allFeaturesSet))
    
    
    
def writeFeatureMatrixToFile(featureMatrix,context_question_list,phase):
    
    n,m = featureMatrix.shape
    
    with open('models/'+phase+".dat","w") as f:
        
        for i in range(n):
            
            qid = context_question_list[i][0]
            score = context_question_list[i][3]
            
            if phase=="train":
                f.write(score+" "+"qid:"+qid)
            else:
                f.write("0 "+"qid:"+qid)
            
            for j in range(m):
                f.write(" "+str(j+1)+":"+str(featureMatrix[i,j]))  
            f.write("\n") 

def trainRanker(training_data_file):
    
    #Each line of training_data_file is of the form - questionID text question score 
    #All questions generated from one sentence have the same questionID  
    #score = correctness, relevance and ambiguity measure
    
    context_question_list = readFile(training_data_file,"train")
    
    
    featureMatrix = FE.extractFeatures(context_question_list)
    
    
    (n,m) = featureMatrix.shape
    
    for i in range(n):
        for j in range(m):
            sys.stdout.write(str(featureMatrix[i,j])+" ")
        print ''
    
    '''
    
    writeFeatureMatrixToFile(featureMatrix,context_question_list,'train')
    

    #train the model using svm-rank
    
    svm_rank_learn_exec_path = "svm_rank/svm_rank_learn"
    feature_vectors_file_path = "models/train.dat"
    model_file_path = "models/model.dat"
    
    command = svm_rank_learn_exec_path+" -c"+" 0.001 "+feature_vectors_file_path+" "+model_file_path
    
    #print command
    
    os.system(command)
    
    #call svm_rank_classify on test data that writes the scores to file
    
    #use the scores to rerank the test output
    '''
  
def testRanker(test_data_file):
    
    context_question_list = readFile(test_data_file,"test")
    
    featureMatrix = FE.extractFeatures(context_question_list)
    
    writeFeatureMatrixToFile(featureMatrix,context_question_list,"test")
    
    svm_rank_classify_exec_path = "svm_rank/svm_rank_classify"
    feature_vectors_file_path = "models/test.dat"
    model_file_path = "models/model.dat"
    output_scores_file_path = "result/prediction.txt"
    
    command = svm_rank_classify_exec_path+" "+feature_vectors_file_path+" "+model_file_path+ " "+output_scores_file_path

    os.system(command)
    
    #return output_scores_file_path
    
#Reads predicted scores from the prediction file and sorts the questions in test file based on those scores.
#The final output is written in the output_file
def rank(test_data_file,output_file):
    
    #to do: rerank among same question ids
    
    prediction_file = 'result/prediction.txt'
    
    scoreList = []
    
    with open(prediction_file,"r") as f:
        for line in f:
            scoreList.append(float(line))
    
    context_question_list = readFile(test_data_file,"test")
    
    questions_by_qid = defaultdict(list)
    
    #append the prediction score at the end
    for idx,item in enumerate(context_question_list):
        item = list(item)
        item.insert(len(item), scoreList[idx])
        item = tuple(item)
        questions_by_qid[item[0]].append(item)
        
    all_qids = questions_by_qid.keys()
    
    sortedQuestions = defaultdict(list)
     
    #sort by increasing prediction score. Less is better
    for qid in all_qids:
        sortedQuestions[qid] = sorted(questions_by_qid[qid],key=lambda question:question[-1])
        
    for qid in all_qids:
        print sortedQuestions[qid]

    with open(output_file,"w") as f:
        for qid in all_qids:
            for item in sortedQuestions[qid]:
                item = list(item)
                line = item[0]+'\t'+item[1]+'\t'+item[2]+'\t'+item[3]+'\n'
                f.write(line)
            f.write("\n")
            
            
            
def getKendallDistance(true,predicted):
    
    kd = 0
    
    n = len(true)
    
    for i in range(0,n):
        for j in range(i+1,n):
            if not((true[i]<true[j] and predicted[i]<predicted[j])or(true[i]>true[j] and predicted[i]>predicted[j])):
                kd+=1
        
    
    kd = float(kd)/(n*(n+1)/2)
    
    return kd

def compute_rank_correlation_metrics(qids_to_ranks_true, qids_to_ranks_predicted):
    
    qids = qids_to_ranks_true.keys()
    
    total_kd_base = 0.0
    total_kd = 0.0
    
    total_spr_base = 0.0
    total_spr = 0.0
    
    for qid in qids:
        listA =  qids_to_ranks_true[qid]
        listB =  qids_to_ranks_predicted[qid]
        #listC = qids_to_ranks_baseline[qid]
        listC = deepcopy(listB)
        random.shuffle(listC)
        
        #print listA 
        #print listC
        
        #kd = getKendallDistance(listA,listB)
        #kd_base = getKendallDistance(listA,listC)
        kd, p_value = kendalltau(listA, listB)
        kd_base,p_value = kendalltau(listA,listC)
        
        spr,p_value = spearmanr(listA,listB)
        spr_base,p_value = spearmanr(listA,listC)
        
        print 'kd = '+str(kd)
        print 'kd_base = '+str(kd_base)
          
        total_kd_base+=kd_base
        total_kd+=kd
        
        total_spr_base+=spr_base
        total_spr+=spr
        
        #print listA
        #print listB
        #print kd_base
        
    avg_kd_base = total_kd_base/len(qids)
    avg_kd = total_kd/len(qids)
    
    avg_spr_base = total_spr_base/len(qids)
    avg_spr = total_spr/len(qids)
    
    print 'average kendall tau distance baseline = %f'%avg_kd_base
    print 'average kendall tau distance = %f'%avg_kd
    
    print 'average kendall tau distance baseline = %f'%avg_spr_base
    print 'average kendall tau distance = %f'%avg_spr
    
    
    return avg_kd_base,avg_kd

            
def evaluate(test_data_file,output_file):
    
    #context_question_list_test = readFile(test_data_file,"test")
    context_question_list_ranked = readFile(output_file,"output") 
    
    #print text_question_list_output 
    
    qids_to_ranks_true = defaultdict(list)
    qids_to_ranks_predicted = defaultdict(list) 
    
    
    for item in context_question_list_ranked:
        item = list(item)
        qid = item[0]
        rank = int(item[-1])
        
        qids_to_ranks_true[qid].append(rank)
        
    for qid in qids_to_ranks_true:
        n = len(qids_to_ranks_true[qid])
        qids_to_ranks_predicted[qid] = range(1,n+1)
    
        
    qids = qids_to_ranks_true.keys()
    
    print qids_to_ranks_true
    print qids_to_ranks_predicted
    
    compute_rank_correlation_metrics(qids_to_ranks_true,qids_to_ranks_predicted)
    
    
   
    
    