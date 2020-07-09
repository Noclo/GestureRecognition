# -*- -*coding: utf-8- #

import os
import time
import numpy as np
import tensorflow as tf
from sklearn import model_selection, utils
from PIL import Image
import cv2
import matplotlib .pyplot as plt
import random
import json  # 保存文件

img_rows = 200
img_cols = 200
img_channels = 1
batch_size = 32
nb_classes = 8 # 测试类别


# 图像保存路径
path = "./"
# 训练的样本路径
path2 = "./img_b"
# 输出
output = ["HEART","GOOD","LOVE","NOTHING","OK","PEACE","PUNCH","STOP"]

def mod_listdir(path):
    # 列出路径下的所有文件名
    # 用来分割训练测试样本
    # 当前目录下的所有文件
    listing = os.listdir(path)
    ret_list = []
    for name in listing:
        if name.startswith('.'):
            continue
        ret_list.append(name)
    return ret_list

def initializer():
    #初始化数据，产生训练测试数据和标签
    img_list = mod_listdir(path2) # 所有图片
    total_images = len(img_list) # 训练样本数量

    # 创建训练样本矩阵
    # PIL 中图像共有9中模式 模式“L”为灰色图像 0黑 255白
    # 转换公式 L = R * 299/1000 + G * 587/1000+ B * 114/1000

    im_matrix = np.array([np.array(Image.open(path2+'/'+image).convert('L'))
                         .flatten() for image in img_list],dtype='float32')
    print(im_matrix.shape)                      # n*(200*200)

    # input("press any key to continue!")

    # 开始创建标签
    label = np.ones((total_images, ), dtype=int)  # 首先全部标记为1
    samples_per_class = total_images / nb_classes # 每类样本数量

    print("sample_per_class - ", samples_per_class)

    s = 0
    r = samples_per_class
    # 开始赋予标签（01234）
    for classIndex in range(nb_classes):
        label[int(s):int(r)] = classIndex
        s = r
        r = s + samples_per_class

    data, label = utils.shuffle(im_matrix, label, random_state=2)
    x = data
    y = label
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.4, random_state=4)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train /= 255 # 黑白 0 1
    x_test /= 255

    return x_train, x_test, y_train, y_test


###开始搭建网络
INPUT_NODE = img_rows * img_cols
OUTPUT_NODE = 8

Image_size = 200
NUM_LABELS = 8

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE= 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE= 3

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 64
CONV3_SIZE= 5

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 64
CONV4_SIZE= 5

FC_SIZE1 = 512
FC_SIZE2 = 128

# 训练用参数
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99  # 权重衰减
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 24000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"


def get_batch(x, y, batch_size):
    # tensorflow 利用batch的思想来加快训练速度
    data = []
    label = []
    m = x.shape[0]
    for _ in range(batch_size):
        index = random.randrange(m) #随机选择一个整数
        data.append(x[index])
        tmp = np.zeros(NUM_LABELS, dtype=np.float32)
        tmp[y[index]] = 1.0
        label.append(tmp)
    return np.array(data), np.array(label) #输出为ndarry

def inference(input_tensor, train, regularizer):
    # 定义前向卷积 添加：dropout 训练有 测试没有
    with tf.name_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, img_channels, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram("w_conv1",conv1_weights)
        tf.summary.histogram("b_conv1", conv1_biases)
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="VALID") # 196*196*32
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding="SAME") # 98*98*32

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram("w_conv2", conv2_weight)
        tf.summary.histogram("b_conv2", conv2_biases)
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1,1,1,1], padding="VALID") # 96*96*64
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") # 48*48*64

    with tf.variable_scope('layer5-conv3'):
        conv3_weight = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias', [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram("w_conv3", conv3_weight)
        tf.summary.histogram("b_conv3", conv3_biases)
        conv3 = tf.nn.conv2d(pool2, conv3_weight, strides=[1,1,1,1], padding="VALID") # 44*44*64
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 22*22*64

    with tf.variable_scope('layer7-conv4'):
        conv4_weight = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('bias', [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram("w_conv4", conv3_weight)
        tf.summary.histogram("b_conv4", conv3_biases)
        conv4 = tf.nn.conv2d(pool3, conv4_weight, strides=[1, 1, 1, 1], padding="VALID")  # 18*18*64
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope('layer8-pool4'):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 9*9*64

    # 然后将第8层的输出变为第9层输入的格式。 后面全连接层需要输入的是向量 将矩阵拉成一个向量
    pool_shape = pool4.get_shape().as_list()
    # pool_shape[0]为一个batch中数据个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool4, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE1], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram("w_fc1", fc1_weights)
        tf.summary.histogram("b_fc1", fc1_biases)
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE1, FC_SIZE2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [FC_SIZE2], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram("w_fc2", fc2_weights)
        tf.summary.histogram("b_fc2", fc2_biases)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [FC_SIZE2, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram("w_fc3", fc3_weights)
        tf.summary.histogram("b_fc3", fc3_biases)
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit       # 这里没有经过softmax ，后面在计算cross_entropy时候利用内置的函数会计算

def test(x_test, y_test):
    with tf.Graph().as_default() as g: # 设置默认graph
        # 定义输入输出格式
        x = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channels], name='x-input')
        y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

        y_ = inference(x, train=None, regularizer=None) # 测试时 不关注正则化损失的值

        # # loss
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y, 1), logits=y_)
        # cross_entropy_mean = tf.reduce_mean(cross_entropy)
        #
        # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 计算总loss
        # tf.summary.scalar("loss", loss)

        # 开始计算正确率
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy",accuracy)

        # 加载模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/", sess.graph)
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 得到迭代轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                for _ in range(x_test.shape[0]):
                    xs, ys = get_batch(x_test, y_test, batch_size=1280) # 测试用
                    rs = sess.run(merged, feed_dict={x: xs, y: ys})
                    writer.add_summary(rs)
                    #print(ys.shape)
                    label, accuracy_score = sess.run([y_, accuracy], feed_dict={x: xs, y: ys})
                    #print("实际手势： %s，  预测手势： %s" % (output[np.argmax(ys)], output[np.argmax(label)]))
                    print("After %s training steps(s), test accuracy = %f" % (global_step, accuracy_score))

            else:
                print("No checkpoint, Training Firstly.")
                return

def decision():
    x_train, x_test, y_train, y_test = initializer()
    test(x_test, y_test)

if __name__ == '__main__':
    decision()













