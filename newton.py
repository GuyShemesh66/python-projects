
import math as math

def newton_raphson(f,f_tag,a):
    i = 0
    eps = 10 ** -13
    xn = (a -f(a)/f_tag(a))
    while (abs(f(xn)) > eps):
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
        xn = (a - f(a) / f_tag(a))
    i = i + 1
    print("x" + str(i) + "is :" + str(xn) + " and f(" + str(xn) + ") is :" + str(ans))
    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn
def f(x):
    return 230 * (x ** 4) + 18 * (x ** 3) + 9 * (x ** 2) - 221 * x - 9
def f_tag(x):
    return 4*230 * (x ** 3) + 3*18 * (x ** 2) + 2*9 * (x ** 1) - 221
a=-0.5
b=0.5
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
print("so the starting point will be  "+str((a+b)/2))
start=((a+b)/2)
newton_raphson(f,f_tag,start)
