# -*- coding:utf-8 -*-
#开发人员：陈禹铭
#开发时间：2020/9/20 12:03
#文件名称：csv_tools.py

import os
import csv
import pandas as pd
import numpy as np

label_list=['cat','dog']

#读取label.list文件  生成 文件名,label 格式的csv文件，方便以后读取文件和对应的label
def create_csv():
    #测试文件夹下读取所有图片和label
    filefold_path='/home/ailab/yxy/dataset/dog_vs_cat/m_train/'
    rows=[]
    f = open("./label/bald_label.list")
    while True:
        line = f.readline()
        if not line:
            break
        line = line.split()
        row = []
        row.append(line[0])
        row.append(line[1])
        rows.append(row)
    csvFile=open("./file_label.csv",'w',encoding='utf-8')
    try:
        writer=csv.writer(csvFile)
        for row in rows:
            writer.writerow(row)
        # for i in range(10):
        #     writer.writerow((i,i+2,i*2))
    finally:
        csvFile.close()

def loadCSVfile(csv_path):
    tmp = pd.DataFrame(pd.read_csv(csv_path))
    data_path = tmp.iloc[:, 0]
    label = tmp.iloc[:, 1]
    return np.array(data_path), np.array(label)

create_csv()
