

import math as math

def triy(func,x):
    i = 0
    eps = 10 ** -13
    xn = x
    while (abs(func(xn))-1 > eps):
        ans = func(xn)
        print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
        i = i + 1
        print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(ans))
        xn = f(xn)
    i=i+1
    print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(ans))
    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn


def f(x):
    return 2-x-x+x**3
a=0.5
b=1
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
print("f(a)*f(b) <0")
triy(f,1.3)