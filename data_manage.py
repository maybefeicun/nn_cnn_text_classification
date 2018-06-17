# -*- coding: utf-8 -*-
# @Time : 2018/6/12 19:21
# @Author : chen
# @Site : 
# @File : data_manage.py
# @Software: PyCharm

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import os
import string
import numpy as np
import pandas as pd
from collections import Counter
import string
import re

data_dir = 'data'

org_train_file = 'train_data.csv'
org_test_file = 'test_data.csv'

# vocab_size = 6000

# with open('stopwords.txt', 'r', encoding='latin-1') as file_read:
#     stopwords = file_read.readlines()
#     stopwords = [stopword.strip() for stopword in stopwords]

# 提取文件中的有用的字段
def userfull_filed(org_file, outuput_file):
    data = pd.read_csv(os.path.join(data_dir, org_file), header=None, encoding='latin-1')
    clf = data.values[:, 0]
    content = data.values[:, -1]
    new_clf = []
    for temp in clf:
        # 这个处理就是将情感评论结果进行所谓的one_hot编码
        if temp == 0:
            new_clf.append([1, 0]) # 消极评论
        # elif temp == 2:
        #     new_clf.append([0, 1, 0]) # 中性评论
        else:
            new_clf.append([0, 1]) # 积极评论

    df = pd.DataFrame(np.c_[new_clf, content], columns=['emotion0', 'emotion1', 'content'])
    df.to_csv(os.path.join(data_dir, outuput_file), index=False)

def sentence_english_manage(line):
    # 英文句子的预处理
    pattern = re.compile(r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789]")
    line = re.sub(pattern, '', line)
    # line = [word for word in line.split() if word not in stopwords]
    return line

def create_lexicon(train_file):
    lemmatizer = WordNetLemmatizer()
    df = pd.read_csv(os.path.join(data_dir, train_file))
    count_word = {} # 统计单词的数量
    all_word = []
    for content in df.values[:, 2]:
        words = word_tokenize(sentence_english_manage(content.lower())) # word_tokenize就是一个分词处理的过程
        for word in words:
            word = lemmatizer.lemmatize(word) # 提取该单词的原型
            all_word.append(word) # 存储所有的单词

    count_word = Counter(all_word)
    # new_word_counter = word_counter.most_common(vocab_size)
    # words, _ = list(zip(*new_word_counter))
    # lex = new_word_counter.keys()
    # if word not in count_word.keys():
    #     count_word[word] = 1
    # else:
    #     count_word[word] += 1

    # count_word = OrderetodDict(sorted(count_word.items(), key=lambda t: t[1]))
    lex = []
    for word in count_word.keys():
        if count_word[word] < 100000 and count_word[word] > 100: # 过滤掉一些单词
            lex.append(word)

    with open('lexcion.pkl', 'wb') as file_write:
        pickle.dump(lex, file_write)

    return lex, count_word

def main():
    # org_file = org_train_file
    # output_file = 'new_train_data.csv'
    # userfull_filed(org_file, output_file)
    # org_file = org_test_file
    # output_file = 'new_test_data.csv'
    # userfull_filed(org_file, output_file)

    create_lexicon('new_train_data.csv')

    df = pd.read_csv(os.path.join(data_dir, 'new_train_data.csv'))
    content_list = df.content
    min_len = min([len(content) for content in content_list])
    max_len = max([len(content) for content in content_list])
    mean_len = np.mean([len(content) for content in content_list])
    print([min_len, max_len, mean_len]) # [6, 374, 74.09011125]

if __name__ == '__main__':
    main()