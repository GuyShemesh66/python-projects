from matplotlib import pyplot as plt
import numpy as np

def ber_mat(N,n):
   return np.random.binomial(n=1, p=0.5, size=(N, n))

def Q_1a(N,n):
    mat = ber_mat(N,n)
    ret = np.mean(mat, axis=1)
    return ret

def empirical_probability(N,n):
 eps=np.linspace(0,1, 50)
 mat=ber_mat(N,n)
 err=np.abs(Q_1a(N,n)-0.5)
 emp = np.zeros(50)
 for i in range(0, 50):
   emp[i]= np.sum(err > eps[i])/N
 return emp

def Q_1bc(N,n):
    eps = np.linspace(0, 1, 50)
    plt.xlabel("epsilon")
    plt.plot(eps, empirical_probability(N,n), "-g",label="empirical probability")
    plt.plot(eps, 2 * np.exp(-2 * n * (eps ** 2)), "-b",label="hoeffding's bound")
    plt.show()



Q_1bc(2000,20)