

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


# 读取文件
POS_CW_PATH = "../../data/shopping_review/pos_cw.txt"
NEG_CW_PATH = "../../data/shopping_review/neg_cw.txt"
pos_data = []
with open(POS_CW_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        pos_data.append(ll)
neg_data = []
with open(NEG_CW_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        neg_data.append(ll)

# 加载字典
stopwords = stopwordslist()

# 删除停止词
POS_STOP_WORD = deleteStop(stopwords, pos_data)
NEG_STOP_WORD = deleteStop(stopwords, neg_data)

# 输出新文件
POS_SW_PATH = "../../data/shopping_review/pos_sw.txt"
NEG_SW_PATH = "../../data/shopping_review/neg_sw.txt"

with open(POS_SW_PATH, "w") as f:
    for item in POS_STOP_WORD:
        f.writelines("%s\n" % ("\t".join(item)))
with open(NEG_SW_PATH, "w") as f:
    for item in NEG_STOP_WORD:
        f.writelines("%s\n" % ("\t".join(item)))
