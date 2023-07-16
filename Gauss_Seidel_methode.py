import numpy as np
import math
from numpy import array, zeros, diag, diagflat, dot
import matplotlib.pyplot as plt

def Gauss_seidel(A,b,N,eps):
    xn = zeros(len(A[0]))
    D = diag(A)
    U = np.triu(A)- diagflat(D)
    L=np.tril(A)- diagflat(D)
    L_D=L+ diagflat(D)
    print(str("A IS"))
    print(str(A))
    print(str("THE FIRST GUSS OF X IS"))
    print(str(xn))
    print(str("D IS"))
    print(str(diagflat(D)))
    print(str("(L+D)^-1 IS"))
    print(str(np.linalg. inv(L_D)))
    print(str("U IS"))
    print(str(U))
    xn1 = dot(np.linalg.inv(L_D), (b - dot(U, xn)))
    plt.xlabel("iterations")
    plt.ylabel("||xn1-xn||1")
    plt.ylim(0,0.05)
    plt.xlim(0, 100)
    i=1
    while np.linalg.norm(xn1-xn,1)> eps and i < N:
        plt.plot(i, np.linalg.norm(xn1 - xn, 1), marker="o",markersize=2, color='black')
        xn = xn1
        np.linalg.norm(xn1-xn,1)
        i = i + 1
        xn1 = dot(np.linalg.inv(L_D), (b - dot(U, xn)))
    print(str("X IS"))
    print(str(xn1[0]))
    print(str("the result got by " + str(i) + "   iterations"))
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
a=mat(n)
b =ans(n)
eps1=10**-16
eps=math.sqrt(np.linalg.cond(a,np.inf)*(((n+1)/np.linalg.norm(a,np.inf))*(n**2)*eps1))
print(str(np.linalg.cond(a,np.inf)*(((n+1)/np.linalg.norm(a,np.inf))*(n**2)*eps1)))
print(str("and the square root "))
print(str(eps))
x=Gauss_seidel(a,b,n**2,eps)
print (str("test"))
print (str(dot(a,x)))
