
def bisection(func,a,b,eps):
    j=0
    if func(a)<0:
        c=a
        a=b
        b=c
    max_error=abs((b-a) / 2)
    change=0.5
    mid=(a+b)/2
    while(abs(func(mid))>eps):
        mid = (a + b) / 2
        res=func(mid)
        print("interaction number is :" + str(j + 1) + ", a is : " + str(a) + ", b is : " + str(b) + ",mid is : " \
              + str(mid) + " ,f(mid) is : " + str(res) + ",max error is :+/- " + str(abs(max_error)))
        if res>0:
            a=mid
        elif res<0:
            b=mid
        else:
            break
        j=j+1
        max_error=max_error*change
    print("the root of the function is  " + str(mid)+ "  by  " + str(j)+" interactions" )
    return mid

def f(x):
 return x**(4)-2

a=1.0
b=4.0
eps=10**-5
print("for a=" + str(a) + ", f(a) is : " + str(f(a)))
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
bisection(f,a,b,eps)