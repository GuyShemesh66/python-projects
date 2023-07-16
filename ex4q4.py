
import math as math
def crossing(func,a,b,i):
    j=0
    max_error=(a - b) / 2
    change=0.5


    while(j<i):
        mid = (a + b) / 2
        res=func(mid)
        print("interaction number is :" + str(j + 1) + ", a is : " + str(a) + ", b is : " + str(b) + ",mid is : " \
              + str(mid) + " ,f(mid) is : " + str(res) + ",max error is :+/- " + str(abs(max_error)))
        if res>0:
            a=mid
        elif res<0:
            b=mid
        else:
            print ("the root of the function is"+str(mid))
            break;
        j=j+1
        max_error=max_error*change
    return mid

def f(x):
 return x**2-x-1

print("for a=-1, f(a) is : " + str(f(-1)) )
print("for b=0, f(b) is : " + str(f(0)))
print("f(a)*f(b) <0")
a = -1.0
b = 0.0

my_ans=crossing(f,a,b,5)