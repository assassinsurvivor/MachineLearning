import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
style.use("fivethirtyeight")


d={'red':[[1,3.7],[2,4.3],[3,4.4]],'green':[[7,5.8],[8,4.9],[9.5,5.7]]}

new_feature=[9.9,5.78]


[[plt.scatter(j[0],j[1],s=100,c=i) for j in d[i]]for i in d]
plt.scatter(new_feature[0],new_feature[1],s=200,c="yellow")

plt.show()

