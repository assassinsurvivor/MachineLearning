import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("fivethirtyeight")

x=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[[3],[5],[9],[9],[11],[13],[16],[17],[19],[21]]

X=np.array(x)
Y=np.array(y)

learning_rate=0.015


m=1
c=0
gues=[]

for i in range(len(x)):

    guess=m*x[i][0]+c
    error=guess-y[i][0]
    m=m-(error)*x[i][0]*learning_rate
    c=c-(error)*learning_rate
    gues.append([guess])
t=np.array(gues)

result=m*X+c

plt.scatter(X,Y)
plt.plot(X,t)#Evolution of the line
plt.plot(X,result,c="green")#The best fit





from sklearn.linear_model import LinearRegression
var=LinearRegression()
var.fit(X,Y)
plt.scatter(X,Y)
plt.plot(X,var.predict(X),c="black")#sklearn's regression fit
plt.show()
