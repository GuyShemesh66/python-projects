#################################
# Your name:Guy Shemesh
#################################



import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.special import expit

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w=np.zeros(data.shape[1])
    threshold = 1e9
    for t in range (T):
        i = np.random.randint(1,data.shape[0],1)[0]
        eta_t = eta_0/(t+1)
        x_i = data[i]
        y_i = labels[i]
        if np.max(w) > threshold:
            w /= threshold
        if np.dot(y_i*w,x_i) < 1:
            w = (1-eta_t)*w+eta_t*C*y_i*x_i
        else:
            w = (1-eta_t)*w
    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w = np.zeros(data.shape[1])
    for t in range(T):
        i = np.random.randint(1, data.shape[0], 1)[0]
        x_i = data[i]
        y_i = labels[i]
        eta_t = eta_0 / (t + 1)
        grad = log_loss_gradient(w, x_i, y_i)
        w =w - eta_t * grad
    return w




#################################
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
def Q1_a():
    eta_0 = [10 ** i for i in np.arange(-6, 5.5, 0.5)]
    runs_num=10
    T=1000
    C=1
    avg_acc_validation=[]
    for eta in eta_0:
      acc_sum=0
      for j in range (runs_num):
          w = SGD_hinge(train_data, train_labels,C, eta,T)
          acc = cal_acc(validation_data, validation_labels, w)
          acc_sum += acc
      avg_acc_validation.append(acc_sum / runs_num)
    plt.xlabel('eta_0')
    plt.ylabel('average accuracy of the validation')
    plt.xscale('log')
    plt.plot(eta_0,avg_acc_validation,marker='o')
    plt.show()
    index=np.argmax(avg_acc_validation)
    return avg_acc_validation[index],eta_0[index]


def cal_acc(data, labels, w):
    success_number = 0
    for i in range(data.shape[0]):
        x_i = data[i]
        y_i = labels[i]
        if np.dot(w,x_i) >= 0:
         y = 1
        else:
         y =-1
        if y_i == y:
           success_number += 1
    return success_number / data.shape[0]


def Q1_b(eta_0):
    C = [10 ** i for i in np.arange(-6, 5.5, 0.5)]
    avg_acc_validation= []
    runs_num = 10
    T=1000
    for c in C:
        acc_sum=0
        for i in range(runs_num):
            w = SGD_hinge(train_data, train_labels,c, eta_0,T)
            acc = cal_acc(validation_data, validation_labels, w)
            acc_sum += acc
        avg_acc_validation.append(acc_sum / runs_num)
    plt.xlabel('C')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.plot(C,avg_acc_validation,marker='o')
    plt.show()
    index=np.argmax(avg_acc_validation)
    return avg_acc_validation[index],C[index]

def Q1_c(eta_0,C):
    T=20000
    w = SGD_hinge(train_data, train_labels,C,eta_0,T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.colorbar()
    plt.show()




def Q1_d(eta_0,C):
    T=20000
    w = SGD_hinge(train_data, train_labels,C,eta_0,T)
    best_acc_classifier=cal_acc(test_data, test_labels, w)
    return  best_acc_classifier

def log_loss_gradient(w, x, y):
    p = softmax([-y * np.dot(w, x), 0])
    z = -y * x *(1-p[1])
    return z


def Q2_a():
    eta_0 = [10 ** i for i in np.arange(-6, 5.5, 0.5)]
    runs_num = 10
    T = 1000
    avg_acc_validation = []
    for eta in eta_0:
        acc_sum = 0
        for j in range(runs_num):
            w = SGD_log(train_data, train_labels, eta, T)
            acc = cal_acc(validation_data, validation_labels, w)
            acc_sum += acc
        avg_acc_validation.append(acc_sum / runs_num)
    plt.xlabel('eta_0')
    plt.ylabel('average accuracy of the validation')
    plt.xscale('log')
    plt.plot(eta_0,avg_acc_validation,marker='o')
    plt.show()
    index=np.argmax(avg_acc_validation)
    return avg_acc_validation[index],eta_0[index]


def Q2_b(eta_0):
    T = 20000
    w = SGD_log(train_data, train_labels, eta_0, T)
    plt.imshow(np.reshape(w,(28, 28)), interpolation='nearest')
    plt.colorbar()
    plt.show()
    test_y = np.sign(np.dot(test_data, w))
    acc = np.mean(test_y == test_labels)
    return acc

def Q2_c(eta_0):
    norms = []
    T=20000
    w = np.zeros(train_data.shape[1])
    for t in range(T):
        i = np.random.randint(1, train_data.shape[0], 1)[0]
        x_i = train_data[i]
        y_i =  train_labels[i]
        eta_t = eta_0 / (t + 1)
        grad = log_loss_gradient(w, x_i, y_i)
        w = w - eta_t * grad
        norms.append(np.linalg.norm(w))
    plt.plot(norms)
    plt.xlabel("Iteration")
    plt.ylabel("Norm of w")
    plt.show()
#################################

accuracy_eta,best_eta=Q1_a()
print(best_eta)
print(accuracy_eta)
accuracy_C,best_C=Q1_b(best_eta)
print(best_C)
print(accuracy_C)
Q1_c(best_eta,best_C)
best_acc_classifier=Q1_d(best_eta,best_C)
print(best_acc_classifier)
acc_eta1, best_eta1=Q2_a()
print(best_eta1)
print(Q2_b(best_eta1))
Q2_c(best_eta1)
