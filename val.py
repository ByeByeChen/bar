import numpy as np
import black_box as blb
from keras.preprocessing import image
import computing_class as cc
import scipy.misc
import _thread
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
test_label_path = "/home/ailab/yxy/script/create_label/val.label"
weights_path = "weights/w/10_weight.npy"
top3_dict = {}
labels = []
W = []
num_classes = 102

def main():
    loadDict()
    loadModel()
    validateByLabelList(test_label_path)
    #label, score = validate("test.jpg")
    #test_thread()

def pr():
    print(1111)

def test_thread():
    while True:
        try:
            _thread.start_new_thread(validate("test,jpg"))
            #_thread.start_new_thread(pr())
        except:
            print("Error:无法启动线程")
        time.sleep(5)

def loadDict():
    global labels
    global top3_dict
    global num_classes
    labels = list(range(0,102))
    print(labels)
    top3_dict = np.load("top3_dict.npy", allow_pickle=True).item()
    print(top3_dict)
    return

def loadModel():
    global W
    W = np.load(weights_path)
    return

def validateByLabelList(test_label_path):
    f = open(test_label_path)
    list = []
    right = 0
    wrong = 0
    right_list = []
    wrong_list = []

    while True:
        line = f.readline()
        if not line:
            break
        line = line.split()
        img_path = line[0]
        label = line[1]
        list.append([img_path, label])
    for arr in list:
        max_label, max_score = validate(arr[0])
        print(max_label, max_score)
        if max_label == int(arr[1]):
            right += 1
            right_list.append(arr[0])
        else:
            wrong += 1
            wrong_list.append(arr[0])
    print("right:",right)
    print("wrong:",wrong)
    print("accuracy:", right / (len(list)))

def validate(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    X = image.img_to_array(img)
    X = np.pad(X, ((12, 12), (12, 12), (0, 0)), 'constant')
    M = cc.get_M()
    P = cc.get_P(W, M)
    X = X + P
    #scipy.misc.imsave('out.jpg', Xi)
    preds = blb.prediction_by_narry(X)
    #print(preds)
    max_score = -1
    max_label = -1
    for i in labels:
        indexs = top3_dict[i]
        sum = 0
        sum1 = 0
        for j in range(len(indexs)):
            sum += preds[0][indexs[j]]
        if max_score < sum:
            max_score = sum
            max_label = i
    return max_label, max_score


if __name__ == "__main__":
    main()

