
# 统计词频


def countWord(data):
    res_dic = {}  # 用字典存储
    for item in data:
        if item in res_dic:
            res_dic[item] = res_dic[item] + 1
        else:
            res_dic[item] = 1
    return res_dic

# 读取文件
POS_SW_PATH = "../../data/shopping_review/pos_sw.txt"
NEG_SW_PATH = "../../data/shopping_review/neg_sw.txt"
pos_data = []
with open(POS_SW_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        pos_data.append(ll)
neg_data = []
with open(NEG_SW_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        neg_data.append(ll)

# 由于要计算好评差评文本总体的词频差异，所以把单类文本进行汇总
pos_all = []
for item in pos_data:
    for word in item:
        pos_all.append(word)
neg_all = []
for item in neg_data:
    for word in item:
        neg_all.append(word)

# 然后就可以开始进行词频统计
pos_dic = countWord(pos_all)
neg_dic = countWord(neg_all)

# 输出词频前20名的词汇
pos_dic = sorted(pos_dic.items(),key = lambda x:x[1], reverse = True)
print("    ".join(["%s: %s" % (item[0], item[1]) for item in pos_dic[0:20]]))
neg_dic = sorted(neg_dic.items(),key = lambda x:x[1], reverse = True)
print("    ".join(["%s: %s" % (item[0], item[1]) for item in neg_dic[0:20]]))
