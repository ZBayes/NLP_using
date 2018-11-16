import sys

sys.path.append("../data_reader")
import reader_tool

content = reader_tool.source_read("../data/THUCNews/体育/1.txt")

# # jieba test
# import jieba
# print('/ '.join(jieba.cut(content)))
# print('\n')
# print('\n')

# print('/ '.join(jieba.cut(content,HMM=False)))
# print('\n')

# print('/ '.join(jieba.cut(content,cut_all=True)))
# print('\n')

# # snownlp test
# from snownlp import SnowNLP

# s = SnowNLP(content)

# print(s.words)
# print(s.tags)
# print(s.sentiments)
# print(s.keywords(5))
# print(s.summary(5))

# 