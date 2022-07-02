import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def distance_generation():

    num=30
    data=np.random.rand(num,2)*100
    x=data[:,0]
    y=data[:,1]
    temp_list=[]
    for i in range(num):
        temp_dict = {}
        temp_dict['x']=x[i]
        temp_dict['y']=y[i]
        temp_list.append(temp_dict)
    df = pd.DataFrame(temp_list)
    df.to_excel('data_generation.xlsx',index=False)
    print(df)

    plt.figure()
    plt.scatter(x,y)
    plt.show()
    distance_matrix=np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            distance_matrix[i,j]=((x[i]-x[j])**2+(y[i]-y[j])**2)**0.5
    return distance_matrix,x,y
#print(distance_matrix)
#distance_matrix,x,y=distance_generation()
#print(distance_matrix)