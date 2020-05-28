# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

data = pd.read_csv('univariateLRData.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

theeta0 = 0
theeta1 = 0

a = 0.0001  
iters = 1000  

n = float(len(X)) 
result_array = np.zeros(iters)

# Performing Gradient Descent 
for i in range(iters): 
    Y_pred = theeta0 + theeta1*X  
    diff=Y_pred-Y
    cost=(1/2*n) * sum(diff * diff)  
    result_array[i] = cost
    D_m = (1/n) * sum(X * (diff)) 
    D_c = (1/n) * sum(diff)  
    theeta1 = theeta1 - a * D_m  
    theeta0 = theeta0 - a * D_c  
print (theeta0, theeta1)
print("------------------------")
#result_array=np.flip(result_array,None)
#print(result_array.reshape(-1,1))

fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylabel('Cost Function')
ax.set_xlabel('Iterations')

_=ax.plot(range(iters),result_array,'b.')









