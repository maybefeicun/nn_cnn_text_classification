# -*- coding: utf-8 -*-
# @Time : 2018/6/15 20:48
# @Author : chen
# @Site : 
# @File : predict.py
# @Software: PyCharm

'''
加载已经训练好的模型
'''

import tensorflow as tf
import pickle
import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# from train_nn import neural_network

with open('lexcion.pkl', 'rb') as file_read:
    lex = pickle.load(file_read)

n_input_layer = len(lex) # 输入层的长度
n_layer_1 = 1500 # 有两个隐藏层
n_layer_2 = 1500
n_output_layer = 2 # 输出层的大小

# X = tf.placeholder('float')
X = tf.placeholder(shape=(None, len(lex)), dtype=tf.float32, name="X")
# Y = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="Y")
# batch_size = 500
# dropout_keep_prob = tf.placeholder(tf.float32)

'''
自己写吧
'''


def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.sigmoid(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.sigmoid(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output

def predict(data):
    output = neural_network(X)
    result = tf.argmax(output, 1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess,
                      os.path.join('nn_model', 'model.ckpt'))


        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(data.lower())
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1.0

        # prediction = sess.run(tf.argmax(output,1),
        #                       feed_dict={X: [features]})
        # prediction = sess.run(tf.argmax(output.eval(feed_dict={X: [features]}), 1))
        _, prediction = sess.run([output, result],
                                 feed_dict={X: [features]})
        print(prediction)

def main():
    predict("i am a bitch, you are stupid, get out !")

if __name__ == '__main__':
    main()