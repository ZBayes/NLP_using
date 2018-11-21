# ---------------------------
# 通过word2vec取均值的方式进行转化，后续运用机器学习的方式进行计算
# STAGE: developing
# TODO:
# [ ] 必要的日志记录
# [ ] 性能记录整理工具
# ---------------------------

import numpy as np
import pandas as pd

import jieba
from gensim.models.word2vec import Word2Vec

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.svm import SVC

import sys
sys.path.append("../")

from data_process import data_loader

# 对每个句子的所有词向量取均值
def buildWordVector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 超参数
N_DIM = 300                         # word2vec的数量
MIN_COUNT = 10                      # 保证出现的词数足够做才进入词典
w2v_EPOCH = 15                      # w2v的训练迭代次数



# 数据加载
pos = data_loader("../../../data/shopping_review/pos.xls",
                  file_type='excel', import_type='pd')
neg = data_loader("../../../data/shopping_review/neg.xls",
                  file_type='excel', import_type='pd')



# 分词
def cw(x): return list(jieba.cut(x))

pos['words'] = pos[0].apply(cw)
neg['words'] = neg[0].apply(cw)



# 数据集划分
y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
x_train, x_test, y_train, y_test = train_test_split(
    np.concatenate((pos['words'], neg['words'])), y, test_size=0.2, random_state=20)



# word2vec-> mean doc2vec
imdb_w2v = Word2Vec(size=N_DIM, min_count=MIN_COUNT)
imdb_w2v.build_vocab(x_train)

imdb_w2v.train(x_train, total_examples=len(x_train), epochs=w2v_EPOCH)

train_vecs = np.concatenate(
    [buildWordVector(z, N_DIM, imdb_w2v) for z in x_train])
test_vecs = np.concatenate(
    [buildWordVector(z, N_DIM, imdb_w2v) for z in x_test])

# 分类模型-ML stage
clf = SVC(kernel='rbf', verbose=True, probability=True,
          shrinking=True, max_iter=10)
clf.fit(train_vecs, y_train)

# 模型评估
score = clf.score(test_vecs, y_test)
y_pred = clf.predict(test_vecs)
paraResultItem = {}
paraResultItem['score'] = score
paraResultItem['precision'] = precision_score(y_test, y_pred)
paraResultItem['recall'] = recall_score(y_test, y_pred)
paraResultItem['f1'] = f1_score(y_test, y_pred)
proba = clf.predict_proba(test_vecs)
Y_prob = []
for i in proba:
    Y_prob.append(i[1])
paraResultItem['auc'] = roc_auc_score(y_test, Y_prob)
print(paraResultItem)
