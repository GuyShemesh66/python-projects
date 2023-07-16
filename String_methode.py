
import scipy as scipy
import numpy as numpy
import math as math

def string(f,a,b):
    i = 0
    eps = 10 ** -13
    xn = (a -f(a)*(b-a)/(f(b)-f(a)))
    while (abs(f(xn)-f(b))>eps and (abs(f(xn))>eps)):
       i = i + 1
       print("x" + str(i) + "  is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(f(xn)))
       temp = xn
       xn=(b -f(b)*(xn-b)/(f(xn)-f(b)))
       b=temp

    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn

def f(x):
    return x**3 +x-1

def g(x):
    return x*(math.exp(-x/3))-1.5
a=0.5
b=0.500001
c=0.6
d=0.7
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for c=" + str(c) + ", f(c) is : " + str(f(c)))
string(f,a,d)

