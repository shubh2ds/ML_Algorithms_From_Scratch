from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
xs=np.array([2,3,4,5,6,7,8,9,10])
ys=np.array([20,30,40,50,60,70,80,90,100])
def find_bestfit_slope_intercept(xs,yx):
    m=((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs))
    b=mean(ys)-(m*mean(xs))
    return m,b
print(find_bestfit_slope_intercept(xs,ys))
m,c=find_bestfit_slope_intercept(xs,ys)
y_pred=[(m*x+c)for x in xs]
plt.scatter(xs[:],ys[:],c='g')
plt.plot(xs[:],y_pred[:],c='b')
def square_error(ys_line,ys_org):
    s=sum((ys_line-ys_org)*(ys_line-ys_org))
    return s

ey_reg=square_error(ys,y_pred)
y_mean=[np.mean(ys) for y in ys]
#print(y_mean)
ey_line=square_error(ys,y_mean)
r2_score=1-(ey_reg/ey_line)
plt.xlabel("Experiance")
plt.ylabel("Salary")
print(r2_score)
def predict(x):
    yp=m*x+c
    plt.scatter(x,yp,c='r')
predict(11)
plt.show()
