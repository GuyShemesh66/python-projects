
import math as math

def f(x):
    return math.sqrt(x)-10


def f_tag(x):
    return 0.5*(1/math.sqrt(x))
def newton_raphson(f,f_tag,a):
    i = 0
    eps = 10 ** -13
    xn = (a -f(a)/f_tag(a))
    while (f_tag(xn)!=0):
        ans = f(xn)
        if ans > 0:
            a = xn
        elif ans < 0:
            b = xn
        else:
            print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
            return xn
        i = i + 1
        print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(ans))
        if (f_tag(xn)!=0):
           xn = (a - f(a) / f_tag(a))
    i = i + 1
    print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(f(xn)))
    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn

a=1
b=100
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
print("so the starting point will be  "+str((a+b)/2))
start=((a+b)/2)
newton_raphson(f,f_tag,start)
