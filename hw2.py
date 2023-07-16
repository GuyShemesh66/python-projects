
import math as math
import math as math
def bisection(func,a,b,i):
    j=0
    max_error=(a - b) / 2
    change=0.5
    if func(a)<0:
        c=a
        a=b
        b=c
    while(j<i):
        mid = (a + b) / 2
        res=func(mid)
        print("interaction number is :"+str(j+1)+", a is : "+str(a)+", b is : "+str(b)+",mid is : "\
              +str(mid)+" ,f(mid) is : "+str(res)+",max error is :+/- " + str(abs(max_error)))
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
 return 120 - 12 * math.sqrt(25 + x ** 2 - (10 * x ** 2)\
                             / math.sqrt(144 - x ** 2) + (25 * x ** 2) / (144 - x ** 2)) - 600 / math.sqrt(144 - x ** 2)

a = 4.0
b = 5.0
print("for a=" + str(a) + ", f(a) is : " +str( f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
print("f(a)*f(b) <0")






my_ans=bisection(f,4,5,10)
com_ans=4.2973
print("my answer is : "+str(my_ans))
print("the computer answer is : "+str(4.2973))
print("the absolute error is : "+str(abs(my_ans-com_ans)))
print("the relative error is : "+str(abs(my_ans-com_ans)/com_ans))