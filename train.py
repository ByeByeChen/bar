# -*- coding:utf-8 -*-
#开发人员：陈禹铭
#开发时间：2020/9/19 19:39
#文件名称：train.py

import tensorflow as tf
import label_list_tools as cs
import numpy as np
import computing_class as cc
import static_top as stt
from keras.preprocessing import image

#train_path = "train_csv.csv"
train_path = "file_label.csv"
batch_size = 8
img_size = 200
channel = 3
blb_classes = 1000
num_classes = 2
lr = 0.001
epoch = 50
q = 8
W=[]
W2=[]



# label_list = tf.constant(['yes','no'])  #目标任务物品类别
# label_list_one_hot = tf.one_hot(label_list,num_classes)  #one_hot

def load_train(train_path, train_label, img_size, classes):
    """
    因为系统内存无法存储那么大的图像矩阵，只能一个batch地去读取图片
    :param train_path: 经过batch_index 规定好范围的图片路径
    :param train_label: 图片label
    :param img_size: 默认224
    :param classes: 有多少个类
    :return: [batch, img_size, img_size, channel]的图像， [batch, num_classes]的label（经过one_hot）
    """
    images = []
    labels_one_hot = []
    for i in range(len(train_path)):
        img= image.load_img(train_path[i], target_size=(img_size, img_size))
        x = image.img_to_array(img)
        x = np.pad(x, ((12,12), (12,12), (0,0)), 'constant')
        images.append(x)
        label = np.zeros(classes)
        label[train_label[i]] = 1.0
        labels_one_hot.append(label)

    images = np.array(images)
    labels_one_hot = np.array(labels_one_hot)

    return images, train_label,labels_one_hot

def training_pro():
    times = 0
    image_path, image_labels = cs.loadCSVfile(train_path)
    top3_dict = stt.get_top3_mapping()
    at = 0.0000001
    batch_index = []
    uj_list=[]
    W = cc.get_W(224,224,3)
    M = cc.get_M()

    for u in range(q):
        uj_list.append(cc.get_U(224,224))
    #print(uj_list)

    # 将 训练数据 分batch
    for i in range(image_path.shape[0]):
        if i % batch_size == 0:
            batch_index.append(i)
    if batch_index[-1] is not image_path.shape[0]:
        batch_index.append(image_path.shape[0])

    i = 0
    while True:
        times = times + 1
        for step in range(len(batch_index) - 1):
            i += 1
            x_list,labels,labels_one_hot = load_train(image_path[batch_index[step]:batch_index[step + 1]],
                                    image_labels[batch_index[step]:batch_index[step + 1]], img_size, num_classes)
            P = cc.get_P(W, M)
            x_p_list = cc.get_X_P_list(x_list,P)
            g_avg = cc.function_g_avg(x_list,x_p_list,W,M,labels,labels_one_hot,batch_size,num_classes,top3_dict,q)
            W = W-at*g_avg
            print("------------------------------------")
            #print(g_avg)
            print("------------------------------------")
        print("times:", times)
        if times % 10 == 0:
            np.save("./weights/w/"+str(times)+"_weight", W, allow_pickle=True, fix_imports=True)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~第",times,"轮权重W已经保存。~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

if __name__ == "__main__":
    training_pro()
