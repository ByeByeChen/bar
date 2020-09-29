import numpy as np
import black_box as blb
from keras.preprocessing import image
import computing_class as cc
import scipy.misc

test_label_path = "./label/test_bald_label.list"
weights_path = "weights/w/10_weight.npy"
top3_dict = {}
labels = []
W = []

def main():
    loadDict()
    loadModel()
    #validateByLabelList(test_label_path)
    label, score = validate("test.jpg")

def loadDict():
    global labels
    global top3_dict
    labels = [0,1]
    top3_dict = np.load("top3_dict.npy", allow_pickle=True).item()
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
    Xi = X + P
    scipy.misc.imsave('out.jpg', Xi)
    preds = blb.prediction_by_narry(X)
    preds1 = blb.prediction_by_narry(Xi)
    print(preds)
    print(preds1)
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

