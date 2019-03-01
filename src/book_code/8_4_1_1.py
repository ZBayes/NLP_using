from sklearn.feature_extraction.text import TfidfVectorizer

# 读取文件
POS_SW_PATH = "../../data/shopping_review/pos_sw.txt"
NEG_SW_PATH = "../../data/shopping_review/neg_sw.txt"
pos_data = []
with open(POS_SW_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        pos_data.append(" ".join(ll))
neg_data = []
with open(NEG_SW_PATH) as f:
    for line in f:
        ll = line.strip().split("\t")
        neg_data.append(" ".join(ll))


all_data = pos_data + neg_data

countVec = TfidfVectorizer(binary=False, norm="l2", decode_error="strict")
res = countVec.fit_transform(all_data)

print(countVec.get_feature_names())
print(res[0])
