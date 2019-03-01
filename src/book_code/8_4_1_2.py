
from gensim.models.word2vec import Word2Vec

POS_SW_PATH = "../../data/shopping_review/pos_sw.txt"
NEG_SW_PATH = "../../data/shopping_review/neg_sw.txt"
N_DIM = 100                         # word2vec的数量
MIN_COUNT = 10                      # 保证出现的词数足够做才进入词典
w2v_EPOCH = 15                      # w2v的训练迭代次数

# 读取文件
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

all_data = pos_data + neg_data
# word2vec
imdb_w2v = Word2Vec(size=N_DIM, min_count=MIN_COUNT)
imdb_w2v.build_vocab(all_data)

imdb_w2v.train(all_data, total_examples=len(all_data), epochs=w2v_EPOCH)

print(imdb_w2v)
print(imdb_w2v.wv.vocab.keys()) # 打印词典中所有词汇
print(imdb_w2v["不错"]) # 查找一个词汇

# 分析相似度
print(imdb_w2v.wv.similarity("极好","气死我了"))
print(imdb_w2v.wv.similarity("别买","讨厌"))
print(imdb_w2v.wv.similar_by_word('极好', topn = 5))
