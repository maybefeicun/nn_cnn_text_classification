# -*- coding: utf-8 -*-
# @Time : 2018/6/12 20:38
# @Author : chen
# @Site : 
# @File : train_nn.py
# @Software: PyCharm

import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import os

with open('lexcion.pkl', 'rb') as file_read:
    lex = pickle.load(file_read)

n_input_layer = len(lex) # 输入层的长度
n_layer_1 = 1500 # 有两个隐藏层
n_layer_2 = 1500
n_output_layer = 2 # 输出层的大小

X = tf.placeholder(shape=(None, len(lex)), dtype=tf.float32, name="X")
Y = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="Y")
batch_size = 500
dropout_keep_prob = tf.placeholder(tf.float32)

def neural_network(data):
    layer_1_w_b = {
        'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
        'b_': tf.Variable(tf.random_normal([n_layer_1]))
    }

    layer_2_w_b = {
        'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
        'b_': tf.Variable(tf.random_normal([n_layer_2]))
    }

    layer_output_w_b = {
        'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
        'b_': tf.Variable(tf.random_normal([n_output_layer]))
    }

    # wx+b
    # 这里有点需要注意那就是最后输出层不需要加激活函数

    full_conn_dropout_1 = tf.nn.dropout(data, dropout_keep_prob)
    layer_1 = tf.add(tf.matmul(full_conn_dropout_1, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.sigmoid(layer_1)
    full_conn_dropout_2 = tf.nn.dropout(layer_1, dropout_keep_prob)
    layer_2 = tf.add(tf.matmul(full_conn_dropout_2, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])
    # layer_output = tf.nn.softmax(layer_output)

    return layer_output

def get_random_n_lines(i, data, batch_size):
    # 从训练集中找训批量训练的数据
    # 这里的逻辑需要理解，同时我们要理解要从积极与消极的两个集合中分层取样
    if ((i * batch_size) % len(data) + batch_size) > len(data):
        rand_index = np.arange(start=((i*batch_size) % len(data)),
                               stop=len(data))
    else:
        rand_index = np.arange(start=((i*batch_size) % len(data)),
                               stop=((i*batch_size) % len(data) + batch_size))

    return data[rand_index, :]

def get_test_data(test_file):
    # 获取测试集的数据用于测试
    lemmatizer = WordNetLemmatizer()
    df = pd.read_csv(os.path.join('data', test_file))
    # groups = df.groupby('emotion1')
    # group_neg_pos = groups.get_group(0).values # 获取非中性评论的信息
    group_neg_pos = df.values

    test_x = group_neg_pos[:, 2]
    test_y = group_neg_pos[:, 0:2]

    new_test_x = []
    for tweet in test_x:
        words = word_tokenize(tweet.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1

        new_test_x.append(features)

    return new_test_x, test_y

def train_neural_network():
    # 配置tensorboard
    tensorboard_dir = "tensorboard/nn"
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # 损失函数
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    tf.summary.scalar("loss", cost_func)
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    # 准确率
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar("accuracy", accuracy)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    df = pd.read_csv(os.path.join('data', 'new_train_data.csv'))
    # data = df.values

    group_by_emotion0 = df.groupby('emotion0')
    group_neg = group_by_emotion0.get_group(0).values
    group_pos = group_by_emotion0.get_group(1).values

    test_x, test_y = get_test_data('new_test_data.csv')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        writer.add_graph(sess.graph)

        lemmatizer = WordNetLemmatizer() # 判断词干所用的
        saver = tf.train.Saver()

        i = 0
        # pre_acc = 0 # 存储前一次的准确率以和后一次的进行比较

        while i < 5000:
            rand_neg_data = get_random_n_lines(i, group_neg, batch_size)
            rand_pos_data = get_random_n_lines(i, group_pos, batch_size)
            rand_data = np.vstack((rand_neg_data, rand_pos_data)) # 矩阵合并
            np.random.shuffle(rand_data) # 打乱顺序

            batch_y = rand_data[:, 0:2] # 获取得分情况
            batch_x = rand_data[:, 2] # 获取内容信息

            new_batch_x = []
            for tweet in batch_x:
                words = word_tokenize(tweet.lower())
                words = [lemmatizer.lemmatize(word) for word in words]

                features = np.zeros(len(lex))
                for word in words:
                    if word in lex:
                        features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
                new_batch_x.append(features)

            # batch_y = group_neg[:, 0: 3] + group_pos[:, 0: 3]

            loss, _, train_acc = sess.run([cost_func, optimizer, accuracy],
                                          feed_dict={X: new_batch_x, Y: batch_y, dropout_keep_prob: 0.6})

            if i % 100 == 0:
                print("第{}次迭代，损失函数为{}, 训练的准确率为{}".format(i, loss, train_acc))
                s = sess.run(merged_summary, feed_dict={X: new_batch_x, Y: batch_y, dropout_keep_prob: 0.6})
                writer.add_summary(s, i)

            if i % 100 == 0:
                # print(sess.run(accuracy, feed_dict={X: new_batch_x, Y: batch_y}))
                test_acc = accuracy.eval({X: test_x[:200], Y: test_y[:200], dropout_keep_prob: 1.0})
                print('测试集的准确率:', test_acc)
                # if test_acc > pre_acc:  # 保存准确率最高的训练模型
                    # print('测试集的准确率: ', test_acc)
                    # pre_acc = test_acc

                    # if not os.path.isdir('./checkpoint'):
                    #     os.mkdir('./checkpoint')
                    # saver.save(sess, './checkpoint/model.ckpt')  # 保存session
                # i = 0
            i += 1

        if not os.path.isdir('./checkpoint'):
            os.mkdir('./checkpoint')
        saver.save(sess, './checkpoint/model.ckpt')  # 保存session

if __name__ == '__main__':
    train_neural_network()
