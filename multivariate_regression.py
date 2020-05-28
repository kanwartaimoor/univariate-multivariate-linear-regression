# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:50:06 2020

@author: Rai Kanwar Taimoor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def Result_Table(table_col1, table_col2):
    table = PrettyTable()
    table.add_column("Actual Label", table_col1)
    table.add_column("Predicted Value", table_col2)
    return table

plt.rcParams['figure.figsize'] = (12.0, 9.0)

data = pd.read_csv('multivariateLRData.csv')
X1 = data.iloc[:, 0]
X2= data.iloc[:, 1]
X3= data.iloc[:, 2]
X4= data.iloc[:, 3]
Y = data.iloc[:, 4]

theeta0 = 0
theeta1 = 0
theeta2 = 0
theeta3 = 0
theeta4 = 0

a = 0.0001  
iters = 1000  

n = float(len(X1)) 
result_array = np.zeros(iters)

# Performing Gradient Descent 
for i in range(iters): 
    Y_pred = theeta0 + theeta1*X1 + theeta2 * X2 + theeta3 * X3 + theeta4 * X4 
    diff=Y_pred-Y
    cost=(1/2*n) * sum(diff * diff)  
       
    result_array[i] =  cost 
    D_1 = (1/n) * sum(diff) 
    D_2 = (1/n) * sum(X1 * (diff)) 
    D_3 = (1/n) * sum(X2 * (diff)) 
    D_4 = (1/n) * sum(X3 * (diff) )
    D_5 = (1/n) * sum(X4 * (diff)) 
   
    
    theeta0 = theeta0 - a * D_1 
    theeta1 = theeta1 - a * D_2 
    theeta2 = theeta2 - a * D_3 
    theeta3 = theeta3 - a * D_4 
    theeta4 = theeta4 - a * D_5 
    
   
print(Result_Table(Y, Y_pred) )
print (theeta0, theeta1)     
    
Y_pred = theeta0 + theeta1*X1 + theeta2 * X2 + theeta3 * X3 + theeta4 * X4 
plt.plot(range(iters),result_array)
