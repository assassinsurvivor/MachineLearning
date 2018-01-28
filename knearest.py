
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import style
import matplotlib.pyplot as plt
style.use("fivethirtyeight")


d={'red':[[1,3.7],[2,4.3],[3,4.4]],'green':[[7,5.8],[8,4.9],[9.5,5.7]]}

new_feature=[9.9,5.78]


[[plt.scatter(j[0],j[1],s=100,c=i) for j in d[i]]for i in d]
plt.scatter(new_feature[0],new_feature[1],s=200,c="yellow")

plt.show()



def knn(data,test,n=3):

    distances=[]

    for classes in data:
        for features in data[classes]:
            modified_euclid_distance=sum(abs(np.array(features)-np.array(new_feature)))#Here instead of using euclidean distance I have used (x2-x1)+(y2-y1) to reduce computational time
            distances.append([modified_euclid_distance,classes])

    
    sorted_distances=sorted(distances)
    vote=[i[1] for i in sorted_distances ]
    confidence=Counter(vote).most_common(1)[0]

    return confidence

print(knn(d,new_feature,3))
            

