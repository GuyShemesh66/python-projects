import numpy as np
from numpy import array, zeros, diag, diagflat, dot
import matplotlib.pyplot as plt

def jacobi(A,b,N,eps):
    xn = zeros(len(A[0]))
    D = diag(A)
    L_U = A - diagflat(D)
    print(str("A IS"))
    print(str(A))
    print(str("THE FIRST GUSS OF X IS"))
    print(str(xn))
    print(str("D IS"))
    print(str(diagflat(D)))
    print(str("L+U IS"))
    print(str(L_U))
    plt.xlabel("iterations")
    plt.ylabel("||xn1-xn||1")
    plt.ylim(0, 1)
    plt.xlim(0,N )
    i=1
    xn1 =dot(np.linalg.inv(diagflat(D)),(b - dot(L_U, xn)))
    while np.linalg.norm(xn1 - xn, 1)>eps and i<N:
        plt.plot(i, np.linalg.norm(xn1 - xn, 1), marker="o",markersize=2, color='black')
        xn=xn1
        i=i+1
        xn1 =dot(np.linalg.inv(diagflat(D)),(b - dot(L_U, xn)))
    print(str("X IS"))
    print(str(xn1[0]))
    print(str("the result got by "+ str(i)+"  iterations"))
    plt.show()
    return xn1[0]

def mat(n):
    a=zeros( (n, n) )
    for i  in range(0,n):
        for j in range(0,n):
            if i!=j:
             a[i][j]=1
            if i==j:
             a[i][j]=n+1
    return a
def ans(n):
    a=zeros( (n, 1) )
    for i  in range(0,n):
        a[i][0]=2*n
    return a

n=50
eps=10**-8
a=mat(n)
print(str(np.linalg.cond(a,np.inf)*(51/np.linalg.norm(a,np.inf)*2500*10**-16)))
b =ans(n)
x=jacobi(a,b,n**2,eps)
print (str("test"))
print (str(dot(a,x)))