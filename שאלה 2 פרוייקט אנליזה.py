import numpy as np

a=10**(-7)
b=10**(-7)
print(str("eps is  ") +str(a) +str("   and delta is ")+str(b))
arr=[[2,b],[a,a*b]]
arr_inv=[[1,-1/(a)],[-1/(b),2/(a*b)]]
def norm_AB(a,b):
    x=(4+(a**2)+(b**2)+(a*b)**2)
    x2=(a*b)**2
    return (((x)+((x**2)-4*(x2))**0.5)/2) ** 0.5

def norm_AB_inv(a,b):
    x = (1 + 1/(b ** 2) + 1/(a ** 2) +4/((a ** 2)*(b ** 2)))
    x2 = 1 / ((a ** 2)*(b**2))
    return (((x) + ((x ** 2) - 4 * (x2)) ** 0.5) / 2) ** 0.5
print(str("eps is  ") +str(a) +str("   and delta is ")+str(b))
print(str("AB"))
print(str(arr))
print(str("(AB)^-1 על פי חישוב "))
print(str(arr_inv))
print(str("(AB)^-1 על פי פונקצית ספריה"))
print(str(np.linalg.inv(arr)))
print(str("חישוב נורמות"))
print(str("חישוב בעזרת פונקציית ספריה לAB"))
print(str(np.linalg.norm(arr,2)))
print(str("חישוב בעזרת ביטוי לנורמה של AB"))
print(str(norm_AB(a,b)))
print(str("חישוב בעזרת פונקציית ספריה לAB^-1"))
x=np.linalg.norm(arr_inv,2)
print(str(x))
print(str("חישוב בעזרת ביטוי לנורמה של AB^-1"))
print(str(norm_AB_inv(a,b)))
print(str("חישוב cond"))
print(str("חישוב בעזרת פונקציית ספריה "))
print(str(np.linalg.cond(arr,2)))
print(str("חישוב בעזרת פונקציית ספריה נורמה כפול נורמה של ההופכי "))
print(str((np.linalg.norm(arr,2))*(np.linalg.norm(arr_inv,2))))
print(str("חישוב בעזרת הביטויים- נורמה כפול נורמה של ההופכי "))
print(str(norm_AB_inv(a,b)*norm_AB(a,b)))