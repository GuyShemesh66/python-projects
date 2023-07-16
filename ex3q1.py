import math
def bisection(func,a,b,eps):
    j=0
    max_error=(a - b) / 2
    change=0.5
    if func(a)<0:
        c=a
        a=b
        b=c
    while(abs(max_error)>eps):
        mid = (b + a) / 2
        res=func(mid)
        print("interaction number is :"+str(j+1)+", a is : "+str(a)+", b is : "+str(b))
        print(",mid is : " +str(mid)+" ,f(mid) is : "+str(res)+",max error is :+/- " + str(abs(max_error)))
        if res>0:
            a=mid
        elif res<0:
            b=mid
        else:
            print ("the root of the function is"+str(mid))
            break
        j=j+1
        max_error=max_error*change
    max_error = max_error * change
    mid = (b + a) / 2
    res = func(mid)
    print("interaction number is :" + str(j + 1) + ", a is : " + str(a) + ", b is : " + str(b))
    print(",mid is : " + str(mid) + " ,f(mid) is : " + str(res) + ",max error is :+/- " + str(abs(max_error)))
    print("The necessary number of iterations is " + str(j + 1) + " to reach a level of accuracy of" + str(abs(eps)))
    return j

def f(x):
    return (x-3)/math.exp(-x/3)
def g(x):
 return ((10**6+(435000/x))*math.exp(x)-(435000/x)-156400)

a=2
b=5
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
print("f(a)*f(b) <0")
eps=10**(-13)
bisection(f,a,b,eps)