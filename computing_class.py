# -*- coding:utf-8 -*-
#开发人员：陈禹铭
#开发时间：2020/9/21 12:42
#文件名称：computing_class.py
import numpy as np
import cupy as cp
import black_box as blb
import math
import cmath
import collections

def get_W(wide_size,height_size,channel_nums):          #产生随机W   224*224*3

    return cp.random.rand(wide_size,height_size,channel_nums)

def get_U(wide_size,height_size,):                  # 产生随机U 向量   224*224

        u_list = cp.random.rand(wide_size,height_size)
        #print(u_list)
        sum =cp.sum(u_list**2)
        sum_sqrt = cp.sqrt(sum)
        u_list = cp.divide(u_list, sum_sqrt)
        #print(u_list)
        u_list = cp.expand_dims(u_list, axis=2)  # 扩维度
        u_list = cp.concatenate((u_list, u_list, u_list), axis=-1)

        return u_list

def get_M():                                    #产生M 向量    224*224*3

        zero = cp.ones((200,200))        #1值   200*200
        one = cp.ones((224,224))         #1值   224*224
        zero=cp.pad(zero,12)             #1值  200*200  填充到  224*224
        m=one-zero                       # 相减
        m = cp.expand_dims(m, axis=2)    #扩维度
        m = cp.concatenate((m, m, m), axis=-1)   #变成 224*224*3
        #print(np.shape(m))
        #print(m[:,14,1])
        return m

def get_P(W,M):                        #产生P 向量
        #W=get_W(224,224,3)
        #M=get_M()
        p=cp.multiply(W,M)
        p = cp.tanh(p)
        #print(np.shape(p))
        #print(p)
        return p * 255

def get_X_P(x,p):
        #p = get_P(w,m)
        #x_p = np.add(x,p)
        x_p = cp.add(x, p)
        return x_p

def get_X_P_list(x_list,p):
        #p = get_P(w,m)
        x_p_list=[]
        for x in x_list:
                x_p = cp.add(x,p)
                x_p_list.append(x_p)
        return x_p_list


#hj（）来表示k-to-1映射函数，该映射函数将一组k个源标签的预测平均作为第j个目标域标签的预测。
# 例如，如果源标签{Tench，Goldfish，Hammerhead}映射到目标标签{ASD}，然后hASD（F（X））=[FTench（X）+FGoldfish（X）+FHammerhead（X）]/3。
# 更一般地说，如果源标签S⊂[K]的子集映射到目标标签j∈[K’]，那么hj（F（X））= 1/(|𝑠|) ∑▒〖𝑠"∈" 𝑆〗  Fs(X），其中| S |是集基数。

def function_h(img,label,top3_dict,top_num):   # h函数
        img = cp.asnumpy(img)
        preds = blb.prediction_by_narry(img)    #调用黑盒检测接口
        indexs = top3_dict[label]            #取出对应label所对应的index列表
        sum = 0
        for i in indexs:
                sum += preds[0][i]            #算总分
                #print(preds[0][i])
        #print("*******************")
        avg = sum/len(indexs)              #平均分
        return avg

def function_h2(img,labels,top3_dict,top_num):   # h函数
        avg_list = []
        img = cp.asnumpy(img)
        preds = blb.prediction_by_narry(img) #调用黑盒检测接口
        for i in labels:
                indexs = top3_dict[i]            #取出对应label所对应的index列表
                sum = 0
                for i in indexs:
                        sum += preds[0][i]            #算总分
                        #print(preds[0][i])
                #print("*****************")
                avg = sum/len(indexs)     #平均分
                if avg == 0:
                    avg = 1.0e-38 
                avg_list.append(avg)
        return avg_list


#  −∑_(𝑖=1)^𝑛 ∑_(𝑗=1)^𝑘′ ωj(1-hj) γ yijloghj(F(Xj+P)) （2）    f函数

def function_f(x_list,x_p_list,labels,labels_one_hot,batch_size,num_classes,top3_dict):

        label_num=[]

        #
        label_num.append(collections.Counter(labels)[0])   #统计0类别的数量
        label_num.append(collections.Counter(labels)[1])   #统计1类别的数量
        ##


        sum = 0
        for i in range(len(x_list)):
                xl = x_list[i].copy()
                xpl = x_p_list[i].copy()
                avg_list_x = function_h2(xl,[0,1],top3_dict,3)
                avg_list_x_p = function_h2(xpl,[0,1],top3_dict,3)
                for j in range(num_classes):
                        yij = labels_one_hot[i][j]
                        if yij == 0:
                            continue
                        w = 1/label_num[j]
                        e = (1-avg_list_x[j])**2
                        #loghj = np.log10(avg_list_x_p[j])
                        loghj = math.log(avg_list_x_p[j])
                        #print("-------------------------------------------")
                        #print("w", w)
                        #print("e", e)
                        #print("loghj", loghj)
                        sum += w*e*yij*loghj
                        #print("sum", sum)
                        #print("-------------------------------------------")

        return -sum

#设f（W）为（2）中定义的损耗，W为优化变量。为了估计梯度∇f（W），我们通过q随机向量扰动使用单边平均梯度估计量（Liu et al.，2018；Tu et al.，2019），其定义为
# ¯𝑔(W)=1/𝑞 ∑_(𝑗=1)^𝑞▒𝑔_𝑗       (3)
#其中{gj}qj=1是形式（4）的q无关随机梯度估计
# gj=b*((f(W+𝛽𝑈𝑗)-f(W))/𝛽)*Uj       (4)
# 其中b是估计量的标量平衡偏差和方差权衡，W∈Rd是向量形式的优化变量集，β是平滑参数，Uj∈Rd是从单位欧几里得球面上均匀随机抽取的向量。

def function_g_avg(x_list,x_p_list1,W,M,labels,labels_one_hot,batch_size,num_classes,top3_dict,q):
        #B=0.01
        b=224
        B = 1 / b
        f1 = function_f(x_list,x_p_list1,labels,labels_one_hot,batch_size,num_classes,top3_dict)
        g_sum = cp.zeros((224,224,3))
        for j in range(q):
                x_p_list2=[]
                uj=get_U(224,224)
                #print(uj)
                #print("----------uj----------")
                W2=cp.add(W, cp.multiply(uj, B))
                #print(W)
                #print("---------W----------")
                p=get_P(W2,M)
                #print(p)
                #print("---------P-----------")
                for x in x_list:
                        x_p_2=get_X_P(x,p)
                        x_p_list2.append(x_p_2)
                #print(x_p_list2)
                f2 = function_f(x_list,x_p_list2,labels,labels_one_hot,batch_size,num_classes,top3_dict)
                print("~~~~~~~~~~~~~~~~~~~f1损失:",f1,",f2损失:",f2,"~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                g = cp.multiply(b*((f2-f1)/B), uj)
                g = cp.array(g)
                #print(g)
                #print("-------G-----")
                g_sum+=g
        return g_sum/q

def function_g_avg2(x_list,x_p_list1,W,M,labels,labels_one_hot,batch_size,num_classes,top3_dict,q,uj_list):
        #B=0.01
        b=224
        B = 1 / b
        f1 = function_f(x_list,x_p_list1,labels,labels_one_hot,batch_size,num_classes,top3_dict)
        g_sum = cp.zeros((224,224,3))
        for j in range(q):
                x_p_list2=[]
                #uj=get_U(224,224)
                W2=W+uj_list[j]*B
                p=get_P(W2,M)
                #print(p)
                for x in x_list:
                        x_p_2=get_X_P(x,p)
                        x_p_list2.append(x_p_2)
                f2 = function_f(x_list,x_p_list2,labels,labels_one_hot,batch_size,num_classes,top3_dict)
                print("~~~~~~~~~~~~~~~~~~~f1损失:",f1,",f2损失:",f2,"~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                g = b*((f2-f1)/B)*uj_list[j]
                g = cp.array(g)
                g_sum+=g
        return g_sum/q

#get_P()
