
import scipy as scipy
import numpy as numpy
import math as math

def string(f,a,b):
    i = 1
    eps = 10 ** -13
    xn = (a -f(a)*(b-a)/(f(b)-f(a)))
    while (abs(f(xn))>eps):
       print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(f(xn)))
       temp = xn
       xn=(b -f(b)*(xn-b)/(f(xn)-f(b)))
       b=temp
       i = i + 1
    print("x" + str(i) + " is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(f(xn)))
    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn

def f(x):
    return 230 * (x ** 4) + 18 * (x ** 3) + 9 * (x ** 2) - 221 * x - 9

def g(x):
    return x*(math.exp(-x/3))-1.5
a=0.9
b=1
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
string(f,a,b)

