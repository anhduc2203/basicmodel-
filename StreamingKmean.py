import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import re
import gensim

from gensim import corpora
from pprint import pprint
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import StreamingKMeans
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

# build dictionary
def build(data):
    word_splitted=[[w for w in content.split()] for content in data]
    return corpora.Dictionary(word_splitted)

# update dictionary
def update(dictionary, update_data):
    update_word_splitted=[[w for w in content.split()] for content in update_data]
    dictionary.add_documents(update_word_splitted)
    return dictionary

# tfidf
def get_tfidf(dictionary):
    tfidf = TfidfVectorizer(
            max_df=0.8, min_df=0.15,
            max_features=None, vocabulary=dictionary.token2id, stop_words=None, encoding='utf8'
        )
    return tfidf

# return matrix tfidf.transform  
def matrix_tfidf(docs, dictionary):
    return get_tfidf(dictionary).fit_transform(docs).toarray()

# Mini Batch Kmean
class MNK(object):
    """
    Find optimal by Mini batch kmean
    """
    def __init__(self, dictionary, data):
        self.data=data
        self.dictionary=dictionary
    
    def get_text(self):
        self.tfidf = TfidfVectorizer(
            max_df=0.8, min_df=0.15,
            max_features=None, vocabulary=self.dictionary.token2id, stop_words=None, encoding='utf8'
        )
        return self.tfidf.fit_transform(self.data)
    
    def find_optimal_clusters(self):
        iters=range(2, 20+1, 1)
        max_clusters=2
        sse=0
        for k in iters:
            mnk = (MiniBatchKMeans(n_clusters=k, init_size=1024,
                                      batch_size=512, random_state=42).fit(self.get_text()))
            # sse.append(mnk.inertia_)
            if (silhouette_score(self.get_text(), mnk.labels_)>sse):
                max_clusters=k
        return max_clusters
    
    def train(self):
        mnk = MiniBatchKMeans(n_clusters=self.find_optimal_clusters(), init_size=1024, 
                              batch_size=512, random_state=42)
        clusters = mnk.fit_predict(self.get_text())
        return mnk, clusters
    
    def get_top_keywords(self):
        mnk, clusters=train()
        self.centroids=mnk.cluster_centers_
        self.labels=mnk.labels_
        self.feature_names=self.tfidf.get_feature_names()
        
        df = pd.DataFrame(self.get_text().todense()).groupby(clusters).mean()
        for i, r in df.iterrows():
            print('\nCluster {0}:'.format(i))
            print(','.join([self.tfidf.get_feature_names()[t] for t in np.argsort(r)[-15:]]))
    
    def get_info(self):
        return self.centroids, self.labels, self.feature_names, self.tfidf

class StreamingUpdate(object):
    """
    Streaming Update: DStream
    """
    def __init__(self, init_clusters, decay_factor, time_unit, sc, ssc):
        self.init_clusters=init_clusters
        self.decay_factor=decay_factor
        self.time_unit=time_unit
        self.sc=sc
        self.ssc=ssc

    # implement
    def streaming(self, mnk, clusters, init_clusters):
        self.mnk=mnk
        self.clusters=clusters
        self.init_clusters=init_clusters
        self.streaming_kmeans=StreamingKMeans(self.init_clusters, self.decay_factor, self.time_unit)
        self.streaming_kmeans.setInitialCenters(self.mnk.cluster_centers_, np.ones([self.init_clusters]))

    # update shape for centers in StreamingContext
    """
    Từ điển được cập nhật khi có tin tức mới đến thì em cập nhật lại kích thước của các centroid
    VD: Từ điển ban đầu có kích thước 10 từ
    Em biểu diễn một câu có 5 từ bằng sparse vector kích thước 5x10
    Từ điển sau khi cập nhật có 15 từ thì câu trên phải biểu diễn lại bằng sparse vector có kích thước 5x15
    Có cách biểu diễn khác mà không phải cập nhật lại biểu diễn của câu không ạ
    """
    def update_shape(self, docs, dictionary):
        self.streaming_kmeans.setRandomCenters(matrix_tfidf(docs, dictionary).shape[1], 1.0, 0)

    # save matrix update
    def save_matrix_update(self, docs, dictionary):
        np.savetxt('/home/ducvu/input_streaming.txt', matrix_tfidf(docs, dictionary))

    # load dstream
    def load_dstream(self):
        self.dstream = self.sc.textFile("/home/ducvu/input_streaming.txt")\
            .map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
        
    # make predict
    def make_predict(self, docs, dictionary):
        self.streaming_kmeans.trainOn(self.load_dstream())
        self.pred_stream=[]
        matrix=matrix_tfidf(docs)
        for x in matrix:
            self.pred_stream.append(self.streaming_kmeans.latest_model.predict(x))
        self.pred_stream=np.array(self.pred_stream)
        
        df = pd.DataFrame(matrix).groupby(self.pred_stream).mean()
        for i, r in df.iterrows():
            print('\nCluster {0}:'.format(i))
            print(','.join([get_tfidf(dictionary).get_feature_names()[t] for t in np.argsort(r)[-15:]]))

if __name__=="__main__":
    # load data
    df=pd.read_csv('/home/ducvu/output05')
    data=list(df['data'][0:2000])
    train, test=data[:1800], data[1800:]
    #build dictionary
    dictionary=build(train)

    #find init number of clusters, matrix feature init
    init_mnk=MNK(dictionary, train)
    mnk, clusters=init_mnk.train()
    init_clusters=init_mnk.find_optimal_clusters()
    matrix_tfidf_init=init_mnk.get_text().toarray()

    # init
    sc=SparkContext("local[2]", "StreamingClustering")
    ssc=StreamingContext(sc, 5) # 5s/batch

    stkm=StreamingUpdate(init_clusters, 1.0, u"batches", sc, ssc)

    """
    Đoạn tin trả về dưới dạng dstream (một tập RDD), e muốn convert dstream này sang string
    để sử dụng tfidf 
    """
    data_streaming=ssc.socketTextStream("localhost", 2203)

    # return dstream
    dstream_data=data_streaming.map(lambda s: list(s))

    words = data_streaming.flatMap(lambda line: line.split(" "))

    stkm.streaming(mnk, clusters, init_clusters)
    stkm.save_matrix_update(dstream_data, dictionary)
    stkm.update_shape(dstream_data, dictionary)
    stkm.load_dstream()
    #
    stkm.make_predict(dstream_data, dictionary)

    ssc.start()
    ssc.awaitTerminationOrTimeout(100)
