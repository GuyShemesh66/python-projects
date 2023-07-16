
import math as math

def false_position(func,a,b):
    i = 0
    eps = 10 ** -14
    if func(a) < 0:
        c = a
        a = b
        b = c
    xn = (a * func(b) - b * func(a)) / (func(b) - func(a))
    while (abs(func(xn)) > eps):
        ans = func(xn)
        if ans > 0:
            a = xn
        if ans < 0:
            b = xn
        i = i + 1
        print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(ans))
        xn = (a * func(b) - b * func(a)) / (func(b) - func(a))
    xn = (a * func(b) - b * func(a)) / (func(b) - func(a))
    i=i+1
    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn


def f(x):
    return 230 * (x ** 4) + 18 * (x ** 3) + 9 * (x ** 2) - 221 * x - 9
a=0.5
b=1
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
print("f(a)*f(b) <0")
false_position(f,a,b)
