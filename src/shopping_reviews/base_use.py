
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys

# 数据加载和分块


def loadfile(is_save=False, test_size=0.2):
    neg = pd.read_excel('../../data/shopping_review/neg.xls',
                        header=None, index=None)
    pos = pd.read_excel('../../data/shopping_review/pos.xls',
                        header=None, index=None)

    def cw(x): return list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(
        np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    if is_save:
        np.save('../../data/shopping_review/y_train.npy', y_train)
        np.save('../../data/shopping_review/y_test.npy', y_test)
    return x_train, y_train, x_test, y_test


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


# 计算词向量
def get_train_vecs(x_train, x_test, n_dim=500, min_count=10, epochs=15):
    # Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=min_count)
    imdb_w2v.build_vocab(x_train)

    imdb_w2v.train(x_train, total_examples=len(x_train), epochs=epochs)

    train_vecs = np.concatenate(
        [buildWordVector(z, n_dim, imdb_w2v) for z in x_train])

    np.save('../../data/shopping_review/train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # Train word2vec on test tweets
    imdb_w2v.train(x_test, total_examples=len(x_test), epochs=epochs)
    imdb_w2v.save('../../data/shopping_review/w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate(
        [buildWordVector(z, n_dim, imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('../../data/shopping_review/test_vecs.npy', test_vecs)
    print(test_vecs.shape)

    return imdb_w2v, train_vecs, test_vecs


def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True, shrinking=False)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, '../../data/shopping_review/svm_model.pkl')
    print(clf.score(test_vecs, y_test))


if __name__ == '__main__':
    # main()
    x_train, y_train, x_test, y_test = loadfile()
    imdb_w2v, train_vecs, test_vecs = get_train_vecs(x_train, x_test)
    svm_train(train_vecs, y_train, test_vecs, y_test)
