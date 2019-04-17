from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import pandas as pd
import pickle
import numpy as np
import os
import re
import time

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


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        test=request.form['comment']
        print("Test before: {0}".format(test))
        test=preprocess(test)
        print("------------------")
        print("Test after: {0}".format(test))
        all_df=pd.read_json('/home/anhduc/Documents/data/text/classify_data/full_data.json')
        documents=all_df['content'].tolist()
        documents.extend([test])
        tfidf=TfidfVectorizer(stop_words=None, max_df=0.7, max_features=20000)
        tfidf_matrix=tfidf.fit_transform(documents)
        max_id=len(documents)-1
        matrix=cosine_similarity(tfidf_matrix[max_id], tfidf_matrix[0:max_id])
        matrix_reshape=matrix.reshape(-1).argsort()
        result=[]
        for i in matrix_reshape[max_id-20:max_id]:
                result.insert(0, documents[i])
        return render_template('result.html', predicts=result)

if __name__=='__main__':
    app.run(debug=True)