from _collections import defaultdict
import featureExtraction as FE
from copy import deepcopy
import os
import sys


def readFile(f):
    
    context_question_list = []
    
    with open(f,"r") as fle:
        for line in fle:
            line = line.rstrip('\n')
            
            if line=="":
                continue
            
            parts = line.split('\t')

            context_question_list.append(tuple(parts))

    return context_question_list


    
def writeFeatureMatrixToFile(featureMatrix,context_question_list,phase):
    
    n,m = featureMatrix.shape
    
    with open('models/'+phase+".dat","w") as f:
        
        for i in range(n):
            
            qid = context_question_list[i][0]
            if phase == 'train':
                score = context_question_list[i][3]
            
            if phase=="train":
                f.write(score+" "+"qid:"+qid)
            else:
                f.write("0 "+"qid:"+qid)
            
            for j in range(m):
                f.write(" "+str(j+1)+":"+str(featureMatrix[i,j]))  
            f.write("\n") 

def trainRanker(training_data_file):
    
    context_question_list = readFile(training_data_file)
    
    
    featureMatrix = FE.extractFeatures(context_question_list)
    
    
    (n,m) = featureMatrix.shape
    
    for i in range(n):
        for j in range(m):
            sys.stdout.write(str(featureMatrix[i,j])+" ")
        print ''
    
    
    writeFeatureMatrixToFile(featureMatrix,context_question_list,'train')
    

    #train the model using svm-rank
    
    svm_rank_learn_exec_path = "svm_rank/svm_rank_learn"
    feature_vectors_file_path = "models/train.dat"
    model_file_path = "models/model.dat"
    
    command = svm_rank_learn_exec_path+" -c"+" 0.001 "+" -t"+" 1 "+feature_vectors_file_path+" "+model_file_path
    #command = svm_rank_learn_exec_path+" -c"+" 0.001 "+feature_vectors_file_path+" "+model_file_path
    #print command
    
    os.system(command)
    
    #call svm_rank_classify on test data that writes the scores to file
    
    #use the scores to rerank the test output

  
def testRanker(test_data_file):
    
    context_question_list = readFile(test_data_file)
    
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
    
    context_question_list = readFile(test_data_file)
    
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
        sortedQuestions[qid] = sorted(questions_by_qid[qid],key=lambda question:question[-1],reverse=True)
        
    for qid in all_qids:
        print sortedQuestions[qid]

    with open(output_file,"w") as f:
        for qid in all_qids:
            for item in sortedQuestions[qid]:
                item = list(item)
                line = ('\t'.join(item[0:-1]))+'\n'
                f.write(line)
            f.write("\n")
            