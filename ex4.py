import scipy as scipy
import numpy as numpy
e=numpy.exp(1)
c=1.4*(10**-9)
better_ans=scipy.integrate.quad(lambda t : e**(c*t), 0, 1)[0]
print("the better anwser is : "+ str(better_ans))
def f(x) :
  e=numpy.exp(1)
  return ((e**(x))-1)/x

k=1.4*(10**-9)
bad_ans=f(k)
print((k**2)/2-(10**-16))
print("the bad anwser is : "+ str(bad_ans))
print ("the relative error is : " + str(abs(better_ans-bad_ans)/better_ans))
