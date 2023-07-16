import numpy as numpy
import math as math
import numpy as numpy

def string_finding(f,a,b):
    return (a -f(a)*(b-a)/(f(b)-f(a)))

def string(f,a,b,eps):
    i = 1
    xn =string_finding(f,a,b)
    while (abs(f(xn))>eps):
       print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(f(xn)))
       temp = xn
       xn=string_finding(f,b,xn)
       b=temp
       i = i + 1
    print("x" + str(i) + " is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(f(xn)))
    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn


def g(x):
    return 230 * (x ** 4) + 18 * (x ** 3) + 9 * (x ** 2) - 221 * x - 9

def f(x):
    return x*math.exp((-x/3))-1.5
a=2
b=2.3
eps=1
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))

string(f,a,b,eps)
