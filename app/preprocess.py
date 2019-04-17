from underthesea import word_tokenize
import pandas as pd
import numpy as np
import re

stopwords = '/home/anhduc/Documents/data/text/vietnamese-stopwords-dash.txt'
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

def preprocess(data):
    # remove stopword
    data=word_tokenize(data, format='text')
    f = open(stopwords, encoding="utf8")
    sw = f.read()
    stop_words = sw.split("\n")
    data=re.sub(r'\d','',data)
    data=re.sub(r'[^\w\s]','',data)
    pre_text = []
    words = data.split()
    for word in words: # remove stopwords
        if word not in stop_words:
            pre_text.append(word)
    data = ' '.join(pre_text)
    return data
if __name__=="__main__":
    s1="adsf"
    s2="qwer"
    p=[]
    p.extend([s1])
    p.extend([s2])
    print(p)