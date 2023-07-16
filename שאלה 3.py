import scipy
def f(n):
    return (n**n)/(scipy.special.factorial(n))

def new_f(n):
    x=1
    for i in range(1, n):
        x=x*(n/(i))
    return x
b=12
print("for b=" + str(b) + ", f(b) is : " + str(f(b)))
print("for b=" + str(b) + ", new_f(b) is : " + str(new_f(b)))

for i in range (1,1001):
    b = i
    print("for b=" + str(b) + ", new_f(b) is : " + str(new_f(b)))


for i in range (1,1001):
    b = i
    print("for b=" + str(b) + ", f(b) is : " + str(f(b)))

