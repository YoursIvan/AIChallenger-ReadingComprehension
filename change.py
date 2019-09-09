import json

from nltk.tokenize import sent_tokenize, word_tokenize
import re
import os
import operator

'''
path = 'C:\\Users\\chendiao\\Desktop\\AIChallenge\\comatch-jieba\\data\\RC\\sequenceAfterWash\\train.json'
dict = {}
dict1 = {}
with open(path, 'r', encoding='utf-8') as fpr:
    data_all = json.load(fpr)
    for instance in data_all:
        temp = []
        text = instance['article']
        article_len = len(text)
        if article_len in dict:
            dict[article_len]  = dict[article_len]+ 1
        else:
            dict[article_len] = 1

        for sentence in text:
            sentence_len = len(sentence)
            if sentence_len in dict1:
                dict1[sentence_len] = dict1[sentence_len] + 1
            else:
                dict1[sentence_len] = 1

sorted_x=sorted(dict.items(),key=operator.itemgetter(0))
sorted_x1=sorted(dict1.items(),key=operator.itemgetter(0))

for key in sorted_x:
    print(key)

for key in sorted_x1:
    print(key)
'''

path = 'C:\\Users\\chendiao\\Desktop\\AIChallenge\\comatch-jieba\\data\\RC\\test.json'
valid1 = 'C:\\Users\\chendiao\\Desktop\\AIChallenge\\comatch-jieba\\data\\RC\\test1.json'
valid2 = 'C:\\Users\\chendiao\\Desktop\\AIChallenge\\comatch-jieba\\data\\RC\\test2.json'



data_names = ['train']
type1 = []
type2 = []
with open(path, 'r', encoding='utf-8') as fpr:
    for line in fpr.readlines():
        data_raw = json.loads(line)
        query = data_raw['query']
        if query[-1] == '吗' or query[-1] == '呀' or query[-1] == '么' or query[-1] == '嘛' or query[-1] == '不' or query[-1] == '吧' or  query[-2:-1] == '咋样' \
                or '是不是' in query or '能不能' in query or '好不好' in query or '算不算' in query or '会不会' in query or '有没有' in query or '要不要' in query or '该不该' in query or '行不行' in query\
                or '需不需要' in query or '可不可以' in query or '需不需要' in query \
                or '是否' in query  or '能否' in query  or '可否' in query or '有无' in query:
            type1.append(data_raw)
        else:
            type2.append(data_raw)

with open(valid1, 'w', encoding='utf-8') as fpw:
    for data_raw in type1:
        json.dump(data_raw, fpw, ensure_ascii=False)
        fpw.write('\n')

with open(valid2, 'w', encoding='utf-8') as fpw:
    for data_raw in type2:
        json.dump(data_raw, fpw, ensure_ascii=False)
        fpw.write('\n')


# with open(valid1, 'w', encoding='utf-8') as fpw:
# #     for data_raw in type1:
# #         fpw.write(data_raw['alternatives'] + '\t' + data_raw['query'] )
# #         fpw.write('\n')
# #
# # with open(valid2, 'w', encoding='utf-8') as fpw:
# #     for data_raw in type2:
# #         fpw.write(data_raw['alternatives'] + '\t' + data_raw['query'])
# #         fpw.write('\n')


