import pandas as pd
import numpy as np
import jieba

# 数据加载
def data_loader(path):
    return pd.read_excel(path, header=None, index=None)

# 分词
def cw(x):
    return list(jieba.cut(x))


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

# 分别进行存储
POS_CW_PATH = "../../data/shopping_review/pos_cw.txt"
NEG_CW_PATH = "../../data/shopping_review/neg_cw.txt"
with open(POS_CW_PATH, "w") as f:
    for item in pos_word_list:
        f.writelines("%s\n" % ("\t".join(item)))
with open(NEG_CW_PATH, "w") as f:
    for item in neg_word_list:
        f.writelines("%s\n" % ("\t".join(item)))
