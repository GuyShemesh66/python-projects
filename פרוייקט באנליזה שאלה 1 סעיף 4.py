
import scipy as scipy
import numpy as numpy
import math as math

def string_complex(f,a,b,eps):
    i = 1
    xn =a -f(a)*(b-a)/(f(b)-f(a))
    while ((abs(f(xn)-f(b))>eps ) and (abs(f(xn))>eps )):
       print("x" + str(i) + "is :"+ str(xn.real)+","+str(xn.imag)+"i" + " and f(" + str(xn) + ") is :" + str(f(xn)))
       temp = xn
       xn=b -f(b)*(xn-b)/(f(xn)-f(b))
       b=temp
       i = i + 1
    print("the root that found is:  (real=" + str(xn.real)+",imag="+str(xn.imag)+"i)" + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn

def f(x):
    return complex(x**3+x-1)
def g(x):
    return complex(x**2+1)

a=complex(-2,0)
b=complex(-0.1,-2)
eps = 10 ** -13
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
string_complex(f,a,b,eps)

a1=complex(-2,0)
b1=complex(-0.1,2)

print(str(f(a1)))
print(str(f(b1)))
print(str(f(a1)*f(b1)))
string_complex(f,a1,b1,eps)