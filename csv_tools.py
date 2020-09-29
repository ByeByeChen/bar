# -*- coding:utf-8 -*-
#开发人员：陈禹铭
#开发时间：2020/9/20 12:03
#文件名称：csv_tools.py

import os
import xml.dom.minidom
import csv
import pandas as pd
import numpy as np

label_list=['blng','dlng']

#方法取xml中label
def get_label_from_xml(path):
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    name = root.getElementsByTagName('name')
    label = name[0].firstChild.data
    return label_list.index(label)

#读取xml文件  生产 文件名,label 格式的csv文件，方便以后读取文件和对应的label
def create_csv():
    #测试文件夹下读取所有图片和label
    filefold_path='E:\darknet检测相关\cameraData\JPEGImages'
    xmlfold_path='E:\darknet检测相关\cameraData\Annotations'
    rows=[]
    file_list = os.listdir(filefold_path)
    xml_list = os.listdir(xmlfold_path)
    #print(file_list)
    #print(label_list)
    for file in file_list:
        row = []
        xml_path = xmlfold_path + '\\'+ file.split('.')[0] + '.xml'
        label = get_label_from_xml(xml_path)
        row.append(file)
        row.append(label)
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
