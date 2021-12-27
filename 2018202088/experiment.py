from nltk import data
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib as plt
import nltk
import re

def init_data():
    '''
    Parameters:None
    Return:X data
           D_Label test
           Phrases_Value a dictionary with value and associated phrase
           MaxLen maximum length in sentences
    '''
    datasetSentences = open("stanfordSentimentTreebank\datasetSentences.txt",'r')
    datasetSplit = open("stanfordSentimentTreebank\datasetSplit.txt",'r')
    sentiment_labels = open("stanfordSentimentTreebank\sentiment_labels.txt",'r')
    dictionary = open("stanfordSentimentTreebank\dictionary.txt",'r')
    datasetSentences.readline()
    datasetSplit.readline()
    sentiment_labels.readline()
    Phrases_Value = {}
    Sen = {}
    Dic = {}
    N = set()
    X = []
    D_Label = []
    Maxlen = 0
    avg = 0
    D = dictionary.readlines()
    S = sentiment_labels.readlines()
    for item in S:
        for i in range(0,len(item)):
            if item[i] == '|':
                Sen[int(item[:i])] = float(item[i+1:])
                break
    for item in D:
        for i in range(0,len(item)):
            if item[i] == '|':
                Dic[int(item[i+1:])] = item[:i]
                break
    for key in Dic.keys():
        Phrases_Value[Dic[key]] = Sen[key]
    for num in range(0,11855):
        x = datasetSentences.readline()
        x = x.lower()
        for i in range(0,len(x)):
            if x[i] == '\t':
                break
        x = x[i+1:]
        x = nltk.WhitespaceTokenizer().tokenize(x)
        for word in x:
            N.add(word)
        X.append(x)
        Maxlen = len(x) if Maxlen < len(x) else Maxlen
        avg += len(x)
        y = datasetSplit.readline()
        for j in range(0,len(y)):
            if y[j] == ',':
                break
        D_Label.append(int(y[j+1]))
    print("l:"+str(avg/len(X))+"\nN:"+str(len(X))+"\n|N|:"+str(len(N)))
    return X,D_Label,Maxlen,Phrases_Value

def word_fre(datesetSenteces):
    res = {}
    for Sentence in datesetSenteces:
        for word in Sentence:
            if word not in res:
                res[word] = 1
            else:
                res[word] += 1
    return res

if __name__ == '__main__':
    X,Y,Maxlen,Phrase_Value = init_data()
    res = word_fre(X)