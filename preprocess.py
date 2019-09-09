#coding=utf-8
import glob
import os
import json
import jieba
import warnings
import re
from itertools import chain

warnings.filterwarnings('ignore')

# build StopWordSet
StopWordsSet = []
with open('StopWords.txt','r',encoding='gb18030') as fp:
    for line in fp.readlines():
        StopWordsSet.append(line.strip('\n'))
    StopWordsSet.append(" ")
    StopWordsSet.append("\"")

def sentence_tokenize(doc):
    sentences = filter(None, re.split('。|！|\!|？|\?',doc))
    return  sentences

def split2char(str):
    charset = []
    for char in str:
        if char not in StopWordsSet:
            charset.append(char)
    return charset


# wd_tokenize for query and passage
def wd_tokenize_pq(text):
    segments_all = jieba.lcut(text, HMM=True, cut_all=False)
    segments = []
    for word in segments_all:
        if word not in StopWordsSet:
            segments.append(word)
    return segments


# wd_tokenize for query and ans
def wd_tokenize(text):
    segments = jieba.lcut(text, HMM=True, cut_all=False)
    return segments

def answer_index(answer,options):
    if answer == options[0]:
        return 0
    elif answer == options[1]:
        return 1
    else:
        return 2
'''
将原始json中的数据分词分句,并且将答案映射成index形式,生成新.json文件存储在data/RC/sequence中
'''

def preprocess(task):
    print('Preprocessing the dataset ' + task + '...')
    data_names = ['train']
    for data_name in data_names:
        data_all = []
        path = os.path.join('data', task, data_name)
        with open(path + '.json', 'r', encoding='utf-8') as fpr:
            for line in fpr.readlines():
                data_raw = json.loads(line)
                instance = {}
                temp = []
                temp_c = []
                instance['q_id'] = data_raw['query_id']
                if data_name != 'test':
                    instance['ground_truth'] = answer_index(data_raw['answer_aw'], data_raw['alternatives_aw'].split('|'))

                flag = 1
                for option in data_raw['alternatives_aw'].split('|'):
                    if re.search('[A-Za-z0-9]',option) != None:
                        flag = 0

                if flag == 1:
                    instance['options'] = [wd_tokenize(option) for option in data_raw['alternatives_aw'].split('|')]
                    for ss in wd_tokenize_pq(re.sub("[A-Za-z0-9]", "", data_raw['query']).strip()):
                        if ss not in list(chain.from_iterable(instance['options'])):
                            temp.append(ss)
                    instance['question'] = temp
                    instance['article'] = [wd_tokenize_pq(s.strip()) for s in
                                           sentence_tokenize(re.sub("[A-Za-z0-9]", "", data_raw['passage']))]


                else:
                    print(data_raw['query_id'])
                    instance['options'] = [wd_tokenize(option) for option in data_raw['alternatives_aw'].split('|')]
                    for ss in wd_tokenize_pq(data_raw['query'].strip()):
                        if ss not in list(chain.from_iterable(instance['options'])):
                            temp.append(ss)
                    instance['question'] = temp
                    instance['article'] = [wd_tokenize_pq(s.strip()) for s in sentence_tokenize(data_raw['passage'])]

                instance['options_c'] = []
                instance['question_c'] = []
                instance['article_c'] = []
                for option in instance['options']:
                    temp = []
                    for word in option:
                        temp.append(split2char(word))
                    instance['options_c'].append(temp)

                for question in instance['question']:
                    instance['question_c'].append(split2char(question))

                for sentence in instance['article']:
                    temp = []
                    for word in sentence:
                        temp.append(split2char(word))
                    instance['article_c'].append(temp)

                data_all.append(instance)
                if len(data_all) % 1000 == 0:
                    print(len(data_all))
        with open(os.path.join('data', task, 'sequenceAfterWash', data_name) + '.json', 'w', encoding='utf-8') as fpw:
        # with open(os.path.join('data', task, 'sequenceAfterWashP', data_name)+'.json', 'w', encoding='utf-8') as fpw:
            json.dump(data_all, fpw,ensure_ascii=False)


def preprocess_vt(task):
    print('Preprocessing the dataset ' + task + '...')
    data_names = ['valid','test']
    for data_name in data_names:
        data_all = []
        path = os.path.join('data', task, data_name)
        with open(path + '.json', 'r', encoding='utf-8') as fpr:
            for line in fpr.readlines():
                data_raw = json.loads(line)
                instance = {}
                temp = []
                instance['q_id'] = data_raw['query_id']
                if data_name != 'test':
                    instance['ground_truth'] = answer_index(data_raw['answer'], data_raw['alternatives'].split('|'))

                flag = 1
                for option in data_raw['alternatives'].split('|'):
                    if re.search('[A-Za-z0-9]',option) != None:
                        flag = 0

                if flag == 1:
                    instance['options'] = [wd_tokenize(option) for option in data_raw['alternatives'].split('|')]
                    for ss in wd_tokenize_pq(re.sub("[A-Za-z0-9]", "", data_raw['query']).strip()):
                        if ss not in list(chain.from_iterable(instance['options'])):
                            temp.append(ss)
                    instance['question'] = temp

                    instance['article'] = [wd_tokenize_pq(s.strip()) for s in
                                           sentence_tokenize(re.sub("[A-Za-z0-9]", "", data_raw['passage']))]

                else:
                    print(data_raw['query_id'])
                    instance['options'] = [wd_tokenize(option) for option in data_raw['alternatives'].split('|')]
                    for ss in wd_tokenize_pq(data_raw['query'].strip()):
                        if ss not in list(chain.from_iterable(instance['options'])):
                            temp.append(ss)
                    instance['question'] = temp
                    instance['article'] = [wd_tokenize_pq(s.strip()) for s in sentence_tokenize(data_raw['passage'])]

                instance['options_c'] = []
                instance['question_c'] = []
                instance['article_c'] = []
                for option in instance['options']:
                    temp = []
                    for word in option:
                        temp.append(split2char(word))
                    instance['options_c'].append(temp)

                for question in instance['question']:
                    instance['question_c'].append(split2char(question))

                for sentence in instance['article']:
                    temp = []
                    for word in sentence:
                        temp.append(split2char(word))
                    instance['article_c'].append(temp)


                data_all.append(instance)
                if len(data_all) % 1000 == 0:
                    print(len(data_all))
        with open(os.path.join('data', task, 'sequenceAfterWash', data_name) + '.json', 'w', encoding='utf-8') as fpw:
        # with open(os.path.join('data', task, 'sequenceAfterWashP', data_name)+'.json', 'w', encoding='utf-8') as fpw:
            json.dump(data_all, fpw,ensure_ascii=False)


if __name__ == '__main__':
    preprocess('RC')
    #preprocess_vt('RC')