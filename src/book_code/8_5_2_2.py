import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

# https://blog.csdn.net/churximi/article/details/61210129


def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)

    return np.array(new_sentences)


# 读入文件路径
POS_PATH = "../../data/shopping_review/pos_sw.txt"
NEG_PATH = "../../data/shopping_review/neg_sw.txt"
N_DIM = 200                         # word2vec的数量
MIN_COUNT = 10                      # 保证出现的词数足够做才进入词典
w2v_EPOCH = 15                      # w2v的训练迭代次数
MAXLEN = 30                         # 句子最大长度

# 读取文件
pos_data = []
with open(POS_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        pos_data.append(ll)
neg_data = []
with open(NEG_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        neg_data.append(ll)

all_data = pos_data + neg_data

# 数据打标签
y = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))

# 数据集打乱
index = [i for i in range(len(all_data))]
np.random.shuffle(index)
x = np.array(all_data)[index]
y = y[index]

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=20)
print("data preprocessing completed")

# word2vec
imdb_w2v = Word2Vec(size=N_DIM, min_count=MIN_COUNT)
imdb_w2v.build_vocab(all_data)
imdb_w2v.train(all_data, total_examples=len(all_data), epochs=w2v_EPOCH)
print("word2vec completed")

# word2vec后处理
n_symbols = len(imdb_w2v.wv.vocab.keys()) + 1
embedding_weights = np.zeros((n_symbols, N_DIM))
idx = 1
word2idx_dic = {}
for w in imdb_w2v.wv.vocab.keys():
    embedding_weights[idx, :] = imdb_w2v[w]
    word2idx_dic[w] = idx
    idx = idx + 1
# print(embedding_weights[0, :])

# 文字转化为数字索引
x_train = text_to_index_array(word2idx_dic, x_train)
x_test = text_to_index_array(word2idx_dic, x_test)
# print(x_train[1])

# 处理为等长
x_train = padded_docs = sequence.pad_sequences(
    x_train, maxlen=MAXLEN, padding='post')
x_test = padded_docs = sequence.pad_sequences(
    x_test, maxlen=MAXLEN, padding='post')


# 开始建立深度学习模型LSTM
model = Sequential()
model.add(Embedding(output_dim=N_DIM,
                    input_dim=n_symbols,
                    weights=[embedding_weights],
                    input_length=MAXLEN, trainable=True))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(output_dim=100,
                             activation='sigmoid',
                             inner_activation='hard_sigmoid', return_sequences=True)))
model.add(Activation('relu'))


model.add(Conv1D(100, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(20))
model.add(Activation('tanh'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, nb_epoch=5,
          validation_data=(x_test, y_test))

print("evaluating...")
score, acc = model.evaluate(x_test, y_test, batch_size=128)
print('Test score: %s ' % score)
print('Test accuracy: %s' % acc)
