import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from util import logTool
from model_ml import model_ml
from model_evaluation import model_report

# 配置值
POS_SW_PATH = "../../../data/shopping_review/pos_pp.txt"
NEG_SW_PATH = "../../../data/shopping_review/neg_pp.txt"
LOG_PATH = "../../../data/shopping_review/log.txt"
MODEL_NUM = "1"

# 日志初始化
log = logTool(LOG_PATH)
log.info('log initiated')

# 数据读取
pos_sw = []
neg_sw = []
with open(POS_SW_PATH) as f:
    for line in f:
        pos_sw.append(line)
with open(NEG_SW_PATH) as f:
    for line in f:
        neg_sw.append(line)
log.info("data imported")

# 打标签
y = np.concatenate((np.ones(len(pos_sw)), np.zeros(len(neg_sw))))

# 数据组合
all_data = np.concatenate((pos_sw, neg_sw))

# TF-IDF处理
countVec = TfidfVectorizer(binary=False, norm="l2", decode_error="strict")
x = countVec.fit_transform(all_data)
log.info("TF-IDF finished")

# 数据集打乱
index = [i for i in range(len(all_data))]
np.random.shuffle(index)
x = x[index]
y = y[index]
log.info("data shuffling finished")

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=20)
log.info("dataset built")

# 分类模型-ML stage
clf, model_msg = model_ml(MODEL_NUM, x_train, y_train)
log.info("model training finished")
log.info("TF-IDF para: binary=False, norm=l2, decode_error=strict")
log.info(model_msg)

# 模型评估
train_result = model_report(clf, x_train, y_train)
test_result = model_report(clf, x_test, y_test)

log.info("train result: %s" % train_result)
log.info("test result: %s" % test_result)