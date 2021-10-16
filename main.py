#import lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read file
path ='data.csv'
data =pd.read_csv(path,header=None,names=['Population', 'Profit'])
#print("data describes\n",data.describe())
#add the ones colum to x data
data.insert(0,'Ones',1)
#print(data)
#get the number of colums
col=data.shape[1]
#seperate X and Y
X=data.iloc[:,0:col-1]
y=data.iloc[:,col-1:col]
#Convert from dataframes to matrix
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))
#plot the points to fit a line into it
#define cost function
def cost_function(X,y,theta):
    z=(1/2*len(X))*np.sum(np.power(((X*theta.T)-y),2))
    return z

#Gradient desecent
def gradientDescent(X,y,theta,alpha,iter):
    m=len(X)
    #save the cost function for each iteration
    cost_history=np.zeros(iter)
    temp=np.matrix(np.zeros(theta.shape))
    #number of parameters is the number of theta
    parameter=int(theta.shape[1])
    for i in range(iter):
        error = (X * theta.T) - y
        #loop through xj (linear regression with multiple variables)
        for j in range(parameter):
            errorterm=np.multiply(error,X[:,j])
            temp[0,j]=temp[0,j]-(alpha/m)*np.sum(errorterm)
        theta=temp
        cost_history[i]=cost_function(X,y,theta)
    return theta,cost_history

#initilize the learning rate and number of iter
alpha = 0.01
iters = 1000
def fit(X,y,theta):
    y_predict=X*theta.T
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(np.ravel(X[:, 1]), np.ravel(y), color="b", marker="o", s=30)
    plt.plot(np.ravel(X[:,1]),np.ravel(y_predict),color='black',linewidth=2,label="prediction")
    plt.show()

#normal Equation to find theta without number of iterations
def normEquation(X,y,theta):
    theta=np.linalg.inv(X.T*X)*X.T*y
    print('Normal Equation')
    print('theta = ', theta)
    print('Cost ', cost_function(X, y, theta.T))



#normEquation(X,y,theta)
g, cost = gradientDescent(X, y, theta, alpha, iters)
print('Gradient Descent Results: ')
print('g = ' , g)
print('cost  = ' , cost[0:5] )
print('Best cost\nCost = ' , cost_function(X, y, g))
print('**************************************')
print('**************************************')
print('**************************************')
fit(X,y,g)
