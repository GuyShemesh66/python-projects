import numpy.random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def Q_2a(tr,lab,img,k):
    distance=[(0,0) for i in range (0,len(tr))]
    for j in range (0,len(tr)):
        distance[j]=(numpy.linalg.norm(tr[j]-img),lab[j])
    distance.sort(key=lambda x: x[0])
    counter = [0 for i in range(10)]
    for i in range(k):
        counter[int(distance[i][1])] += 1
    max_v = 0
    max_in = 0
    for i in range(0, 10):
        if (counter[i] > max_v):
            max_v = counter[i]
            max_in = i
    return max_in

def Q_2b(n,tr,lab,img,k,test_lab):
    counter = 0
    t = tr[:n]
    l = lab[:n]
    for i in range(len(img)):
        ans = Q_2a(t,l, img[i], k)
        if (int(test_lab[i])==ans):
            counter=counter+1
    ret=counter / len(img)
    return ret

print(Q_2b(1000,train,train_labels,test,10,test_labels))

def Q_2c(n,tr,lab,img,k,test_lab):
    y=[0 for i in range (1,k+1)]
    x = [i for i in range(1, k+1)]
    for i in range(0,k):
        y[i]=Q_2b(n,tr,lab,img,i+1,test_lab)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.plot(x,y)
    plt.show()

Q_2c(1000,train,train_labels,test,100,test_labels)

def Q_2d(n,tr,lab,img,test_lab):
    y = [0 for i in range(100, n+1,100)]
    x = [i for i in range(100, n+1,100)]
    for i in range(0,int(n/100)):
        y[i]=Q_2b((i+1)*100,tr,lab,img,1,test_lab)
    plt.xlabel("n")
    plt.ylabel("Accuracy")
    plt.plot(x,y)
    plt.show()




Q_2d(5000,train,train_labels,test,test_labels)