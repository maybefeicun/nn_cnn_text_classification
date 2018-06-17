# -*- coding: utf-8 -*-
# @Time : 2018/6/13 19:14
# @Author : chen
# @Site : 
# @File : train_cnn.py
# @Software: PyCharm

import os
import random
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from train_nn import get_random_n_lines
from data_manage import sentence_english_manage

with open('lexcion.pkl', 'rb') as file_read:
    lex = pickle.load(file_read)

input_size = len(lex) # 输入的长度
num_classes = 2 # 分类的数量
batch_size = 64
seq_length = 150 # 一个tweet的固定长度

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.float32, [None, num_classes])

dropout_keep_prob = tf.placeholder(tf.float32)

def neural_network():
    '''
    整个流程的解释：
    输入为一个X，shape=[None, 8057]，8057为字典的长度
    首先利用embedding_lookup的方法，将X转换为[None, 8057, 128]的向量，但有个疑惑就是emdeding_lookup的实际用法，在ceshi.py中有介绍
    接着expande_dims使结果变成[None, 8057, 128, 1]的向量，但这样做的原因不是很清楚，原因就是通道数的设置

    然后进行卷积与池化：
    卷积核的大小有3种，每种卷积后的feature_map的数量为128
    卷积核的shape=[3/4/5, 128, 1, 128]，其中前两个为卷积核的长宽，最后一个为卷积核的数量，第三个就是通道数
    卷积的结果为[None, 8057-3+1, 1, 128]，矩阵的宽度已经变为1了，这里要注意下

    池化层的大小需要注意：shape=[1, 8055, 1, 1]这样的化池化后的结果为[None, 1, 1, 128]
    以上就是一个典型的文本CNN的过程
    :return:
    '''

    ''' 进行修改采用短编码 '''

    ''' tf.name_scope() 与 tf.variable_scope()的作用基本一致'''
    with tf.name_scope("embedding"):
        embedding_size = 64
        '''
        这里出现了一个问题没有注明上限与下限
        '''
        # embeding = tf.get_variable("embedding", [input_size, embedding_size]) # 词嵌入矩阵
        embedding = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0)) # 词嵌入矩阵
        # with tf.Session() as sess:
        #     # sess.run(tf.initialize_all_variables())
        #     temp = sess.run(embedding)
        embedded_chars = tf.nn.embedding_lookup(embedding, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) # 设置通道数

    # 卷积与池化层
    num_filters = 256 # 卷积核的数量
    filter_sizes = [3, 4, 5] # 卷积核的大小
    pooled_outputs = []

    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv_maxpool_{}".format(filter_size)):
            filter_shape = [filter_size, embedding_size, 1, num_filters] # 要注意下卷积核大小的设置
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b)) # 煞笔忘了加这个偏置的加法

            pooled = tf.nn.max_pool(h, ksize=[1, seq_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)


    num_filters_total = num_filters * len(filter_sizes)
    '''
    # tensor t3 with shape [2, 3]
    # tensor t4 with shape [2, 3]
    tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
    tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
    '''
    h_pool = tf.concat(pooled_outputs, 3) # 原本是一个[None, 1, 1, 128]变成了[None, 1, 1, 384]
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # 拉平处理 [None, 384]

    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # output
    with tf.name_scope("output"):
        # 这里就是最后的一个全连接层处理
        # from tensorflow.contrib.layers import xavier_initializer
        W = tf.get_variable("w", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer()) # 这个初始化要记住
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        output = tf.nn.xw_plus_b(h_drop, W, b)
        # output = tf.nn.relu(output)

    return output

def get_new_batch_x(batch_x):
    lemmatizer = WordNetLemmatizer()

    new_batch_x = []
    for tweet in batch_x:
        # 这段循环的意义就是将单词提取词干，并将字符转换成下标
        words = word_tokenize(sentence_english_manage(tweet.lower()))
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(seq_length)  # 设置长度
        tip = 0
        for word in words:
            if tip >= seq_length:
                break
            if word in lex:
                features[tip] = lex.index(word)  # 使用了新的编码方式
                tip += 1
            # features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        new_batch_x.append(features.T)

    return new_batch_x

def train_neural_netword():
    # 配置tensorboard
    tensorboard_dir = "tensorboard/cnn"
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    output = neural_network()

    # 构建准确率的计算过程
    predictions = tf.argmax(output, 1)
    correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float")) # 强制转换
    tf.summary.scalar("accuracy", accuracy)

    # 构建损失函数的计算过程
    optimizer = tf.train.AdamOptimizer(0.001)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    tf.summary.scalar("loss", loss) # 将损失函数加入
    train_op = optimizer.minimize(loss)
    # grads_and_vars = optimizer.compute_gradients(loss)
    # train_op = optimizer.apply_gradients(grads_and_vars)

    # 将参数保存如tensorboard中
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 构建模型的保存模型
    saver = tf.train.Saver(tf.global_variables())

    # 数据集的获取
    df = pd.read_csv(os.path.join('data', 'new_train_data.csv'))

    group_by_emotion0 = df.groupby('emotion0')
    group_neg = group_by_emotion0.get_group(1).values
    group_pos = group_by_emotion0.get_group(0).values

    df = pd.read_csv(os.path.join('data', 'new_test_data.csv'))
    group_neg_pos = df.values
    test_x = group_neg_pos[:, 2]
    test_y = group_neg_pos[:, 0:2]

    # test_x, test_y = get_test_data('new_test_data.csv')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # sess.run(tf.global_variables_initializer())
        # 将图像加入tensorboard中
        writer.add_graph(sess.graph)

        i = 0
        # pre_acc = 0
        while i < 10000:
            rand_neg_data = get_random_n_lines(i, group_neg, batch_size)
            rand_pos_data = get_random_n_lines(i, group_pos, batch_size)
            rand_data = np.vstack((rand_neg_data, rand_pos_data))
            np.random.shuffle(rand_data)

            batch_x = rand_data[:, 2]
            batch_y = rand_data[:, 0: 2]
            new_batch_x = get_new_batch_x(batch_x)


            _, loss_, train_acc = sess.run([train_op, loss, accuracy], feed_dict={X: new_batch_x, Y: batch_y, dropout_keep_prob: 0.6})

            if i % 50 == 0:
                print("loss:{}\ntrain_acc:{}".format(loss_, train_acc))
                # 每二十次保存一次tensorboard
                s = sess.run(merged_summary, feed_dict={X: new_batch_x, Y: batch_y, dropout_keep_prob: 0.6})
                writer.add_summary(s, i)

            if i % 50 == 0:
                # 每10次打印准确率（这是指评测的准确率）
                new_test_x = get_new_batch_x(test_x)
                accur = sess.run(accuracy, feed_dict={X: new_test_x[: 100], Y: test_y[: 100], dropout_keep_prob: 1.0})
                print("test_acc", accur)
                # if accur > pre_acc:
                #     # 当前的准确率高于之前的准确率，更新模型
                #     pre_acc = accur
                #     print("准确率:", pre_acc)
                #     tf.summary.scalar("accur", accur)
            i += 1

        saver.save(sess, "cnn_model/model.ckpt")


if __name__ == '__main__':
    train_neural_netword()