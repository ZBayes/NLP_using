import pandas as pd
import numpy as np
import jieba

# 数据加载
def data_loader(path):
    return pd.read_excel(path, header=None, index=None)

# 分词
def cw(x):
    return list(jieba.cut(x))

# 创建停用词列表
def stopwordslist():
    STOPWORD_PATH = "../../../data/stopwords/hit.txt"
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
POS_PATH = "../../../data/shopping_review/pos.xls"
NEG_PATH = "../../../data/shopping_review/neg.xls"

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
POS_STOP_WORD = deleteStop(stopwords, pos_word_list)
NEG_STOP_WORD = deleteStop(stopwords, neg_word_list)

# 输出新文件
POS_SW_PATH = "../../../data/shopping_review/pos_pp.txt"
NEG_SW_PATH = "../../../data/shopping_review/neg_pp.txt"

with open(POS_SW_PATH, "w") as f:
    for item in POS_STOP_WORD:
        f.writelines("%s\n" % (" ".join(item)))
with open(NEG_SW_PATH, "w") as f:
    for item in NEG_STOP_WORD:
        f.writelines("%s\n" % (" ".join(item)))

