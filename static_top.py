# -*- coding:utf-8 -*-
#开发人员：陈禹铭
#开发时间：2020/9/20 12:03
#文件名称：static_top.py

import numpy as np
from collections import Counter
import label_list_tools as cs
import black_box as blb

def get_top3_mapping():     #k——to-one 映射

    csv_path = './file_label.csv'   #csv文件路径
    top3_dict={}                    #存储每个label对应的top3映射
    image_path, labels = cs.loadCSVfile(csv_path) #读取csv文件，获取文件名和labels

    for i in range(len(image_path)):        #遍历img
        index_arr = []    #存储每个图片label对应的top3高分的index
        preds = blb.prediction_by_path(image_path[i])  #图片传入黑盒 获取输出preds  维度1000
        arr = np.array(preds)
        arr = np.argsort(-arr)   # 排序 倒序 arr元素变成了index
        index_arr.append(arr[0][0])
        index_arr.append(arr[0][1])
        index_arr.append(arr[0][2])


        if labels[i] in top3_dict:    #判断是否存在标签   存在更新对应值   不存在就新加入
            top3_dict[labels[i]] = top3_dict[labels[i]] + index_arr
        else:
            top3_dict[labels[i]] = index_arr

    for key in top3_dict.keys():         #遍历每个label对应的index集合
        top3 = Counter(top3_dict[key]).most_common(3)   #选出评率最高的3个
        for i in range(len(top3)):
            top3[i] = top3[i][0]
        top3_dict[key] = top3  #用频率最高的3个代替原label对应的index列表
    np.save("top3_dict.npy", top3_dict)


    return top3_dict







# arr_all_index = []
#arr = np.array(arr1)
# np.argsort(arr)
# arr_all_index.append(arr[0])
# arr_all_index.append(arr[1])
# arr_all_index.append(arr[2])
#top3 = Counter(arr_all_index).most_common(3)
# print(arr)
# print(np.sort(arr))#或print np.sort(arr,axis=None)
# print (np.argsort(arr)) # 正序输出索引，从小到大
# print (np.argsort(-arr))
# top3=Counter(arr1+arr2).most_common(3)
# print(top3) #统计数组中出现频率最高的3个

