import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import sigmoid
from PIL import Image
import pickle

#Define SoftMax
def softmax1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

#Block the Overflow!
def softmax2(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# a = np.array([0.3, 2.9, 4.0])
#
# y = softmax2(a)
# print(y)

# exp_a = np.exp(a)
# print(exp_a)
#
# sum_exp_a = np.sum(exp_a)
# print(sum_exp_a)
#
# y = exp_a / sum_exp_a
# print(y)

#Show image!
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)

    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax2(a3)

    return y

def mean_squared_errot(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(x_train.shape)

# --------
# img = x_train[0]
# label = t_train[0]
# print(label)
#
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
#
# img_show(img)

x, t = get_data()
network = init_network()

batch_size = 100 #배치 크기

accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    # y = predict(network, x[i])
    # p = np.argmax(y) #확률이 가장 높은 원소의 인덱스를 얻음
    # if p == t[i]:
    #     accuracy_cnt += 1
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt)/ len(x)))
