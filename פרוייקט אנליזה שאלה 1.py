import random
import numpy
def ex1(func1,func2,a,eps):
    i = 1
    xn = func2(a)
    while ((abs(func1(xn))>eps)and i<1000 ):
        ans = func1(xn)
        if abs(ans)> eps:
            a = xn
        else:
            print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
            return xn
        print("x" + str(i) + "is :" + str(xn) + " and its not the root")
        i = i + 1
        xn = func2(a)
    if(abs(func1(xn))>eps) or i==1000 :
        print("the root not found")
        return
    print("the root that found is:" + str(xn) + " with error rang of: " + str(eps) + " by " + str(i) + " iterations")
    return xn


def f(x):
    return 1*(x**3)-5*(x**2)

def g1(x):
    return x-((x**3 -5*x**2)/25)
def g2(x):
    return ((125-x**2)/(15+x))
def g3(x):
    return (x+1)

def random(a,b):
       x=numpy.random.uniform(a,b)
       return x







a=-10
b=10
eps=10**-8
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
x=random(a,b)
print("the random starting point is  " + str(x) )
ex1(f,g2,x,eps)