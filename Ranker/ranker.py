import sys
import model as md



def main(arguments):
    
    if len(arguments)<2:
        print("usage: rank.py training_data test_data outputFile")
        
    training_data_file = arguments[0]
    test_data_file = arguments[1] 
    output_file = arguments[2]
    
    #md.trainRanker(training_data_file)
    md.testRanker(test_data_file)
    md.rank(test_data_file,output_file)
    md.evaluate(test_data_file,output_file)
    
if __name__ == "__main__":
    
    gold_standard = 'data/gs.txt'
    train = 'data/train.txt'
    #train = 'data/train_sample.txt'
    test = 'data/test.txt'
    #test = 'data/test_sample.txt'
    output = 'result/output.txt'
    
    
    
    #main([train,test,output])