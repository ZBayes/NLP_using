import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 数据加载
def data_loader(path):
    return pd.read_excel(path, header=None, index=None)

# 分词
def cw(x):
    return list(jieba.cut(x))

# 创建停用词列表
def stopwordslist():
    STOPWORD_PATH = "../../data/stopwords/hit.txt"
    stopwords = [line.strip() for line in open(
        STOPWORD_PATH, encoding='GBK').readlines()]
    return stopwords

# 删除停止词
def deleteStop(stopList, data):
    data_new = []
    for item in data:
        item_new = []
        for word in item:
            if word in stopList:
                continue
            item_new.append(word)
        data_new.append(item_new)
    return data_new


# 读入文件路径
POS_PATH = "../../data/shopping_review/pos.xls"
NEG_PATH = "../../data/shopping_review/neg.xls"

# 加载正类和负类文档
pos = data_loader(POS_PATH)
neg = data_loader(NEG_PATH)

# 进行批量化分词
pos['words'] = pos[0].apply(cw)
neg['words'] = neg[0].apply(cw)

# 从pd中取出数据
pos_word_list = []
for item in pos['words']:
    pos_word_list.append(item)
neg_word_list = []
for item in neg['words']:
    neg_word_list.append(item)

# 加载字典
stopwords = stopwordslist()

# 删除停止词
pos_sw = deleteStop(stopwords, pos_word_list)
neg_sw = deleteStop(stopwords, neg_word_list)

# 打标签
y = np.concatenate((np.ones(len(pos_sw)), np.zeros(len(neg_sw))))

# 将句子从list类型链接为以空格隔开的句子
all_data = []
for item in np.concatenate((pos_sw, neg_sw)):
	all_data.append(" ".join(item))
print(len(all_data))
print(all_data[0])

# TF-IDF处理
countVec = TfidfVectorizer(binary=False, norm="l2", decode_error="strict")
x = countVec.fit_transform(all_data)

# 数据集打乱
index = [i for i in range(len(all_data))]
np.random.shuffle(index)
print(index)
x = x[index]
y = y[index]

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=20)

# 分类模型-ML stage
clf = SVC(kernel='rbf', probability=True,
          shrinking=True)
clf.fit(x_train, y_train)

# 模型评估
score = clf.score(x_test, y_test)
y_pred = clf.predict(x_test)
paraResultItem = {}
paraResultItem['score'] = score
paraResultItem['precision'] = precision_score(y_test, y_pred)
paraResultItem['recall'] = recall_score(y_test, y_pred)
paraResultItem['f1'] = f1_score(y_test, y_pred)
proba = clf.predict_proba(x_test)
Y_prob = []
for i in proba:
    Y_prob.append(i[1])
paraResultItem['auc'] = roc_auc_score(y_test, Y_prob)

print("model evalutaion: %s" % paraResultItem)