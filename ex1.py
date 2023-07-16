import scipy as scipy
import numpy as numpy
e=numpy.exp(1)
c=1.4*(10**-9)
print (scipy.integrate.quad(lambda t : e**(c*t), 0, 1))
