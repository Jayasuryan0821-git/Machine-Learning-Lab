#!/usr/bin/env python
# coding: utf-8

# 1. Create the following data set for two independent variable (X1,X2) and one dependent variable (Y) in CSV. Apply the Logistic Regression to perform the following.
# 
# a. Calculate the coefficients (B0, B1 and B2).
# 
# b. Apply the sigmoid function to get the prediction and calculate error.
# 
# c. From the predicted values calculate the accuracy.
# 
# d. List the model parameters along with error for every instance of the training data.
# 
# e. Plot the graph of B1 v/s error and B2 v/s error.
# 
# f. Use scikit learn model to repeat the above steps and compare the results.

# In[12]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import copy 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
df = pd.read_csv('data.csv')
x,y = df[['X1','X2']].values,df['Y'].values

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w,x[i]) + b)
        cost += -y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)  
    cost /= m
    return cost 

def compute_gradient(x,y,w,b):
    m,n = x.shape 
    dj_dw,dj_db = np.zeros(n),0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i],w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i*x[i,j]
        dj_db += err_i
    dj_dw,dj_db = dj_dw/m,dj_db/m
    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,alpha,iters):
    w = copy.deepcopy(w_in)
    b = b_in
    j_hist,b1_val,b2_val = [],[],[]
    for i in range(iters + 1):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w -= alpha*dj_dw
        b1_val.append(w[0])
        b2_val.append(w[1])
        b -= alpha*dj_db
        j_hist.append(compute_cost(x,y,w,b))
        print(f"Iterations {i} / {iters} b1:{w[0]} b2:{w[1]} b0:{b} Cost: {compute_cost(x,y,w,b)}")
    return w,b,j_hist,b1_val,b2_val


w_tmp,b_tmp,alpha,iters = np.zeros(x.shape[1]),0.0,0.1,50
w,b,j_hist,b1_val,b2_val = gradient_descent(x,y,w_tmp,b_tmp,alpha,iters)
print(f"Updated parameters for slopes and intercept:\nb1,b2:{w}\nintercept:{b}\nMSE:{j_hist[-1]}")
        
def compute_accuracy(x,w,b):
    pred,num_correct = [],0
    m = x.shape[0]
    for i in range(m):
        y_pred = sigmoid(np.dot(x[i],w) + b)
        if y_pred > 0.5:
            pred.append(1)
            num_correct += 1
        elif y_pred < 0.5:
            pred.append(0)
            num_correct += 1
    accuracy = num_correct/len(pred)
    return accuracy,pred

accuracy,pred = compute_accuracy(x,w,b)
print(f"For {x} prediction is {pred} and Accuracy:{accuracy}")
plt.subplot(1,2,1)
plt.plot(b1_val,j_hist)
plt.title('B1 values vs errors')
plt.xlabel('B1 values')
plt.ylabel('Error values')
plt.subplot(1,2,2)
plt.plot(b2_val,j_hist)
plt.title('B2 values vs Errors')
plt.xlabel('B2 values')
plt.ylabel('Error')
plt.tight_layout()
plt.show()

model = LogisticRegression(solver='sag')
model.fit(x,y)
y_pred = model.predict(x)
acc = accuracy_score(y,y_pred)
loss = log_loss(y,y_pred)
print(f"Sklearn model slope coeff:{model.coef_} and intercept:{model.intercept_}\nError:{loss}\nAccuracy:{acc}")


# 2. Use above data set for one independent variable (X=X1) and one dependent variable (Y) in CSV. Applying Logistic Regression, explore the relationship between independent and dependent variables.
# 
# a. Calculate the coefficients (B0, and B1).
# 
# b. Apply the sigmoid function to get the prediction and calculate error.
# 
# c. From the predicted values calculate the accuracy.
# 
# d. List the model parameters along with error for every instance of the training data.
# 
# e. Plot the graph of B1 v/s error.
# 
# f. Visualize the following binary cross entropy function for logistic regression using the above training data Plot y=1 and y=0 cases separately, and then plot the combined graph by considering X in X-axis and cost in Y-axis.

# In[17]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import copy 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
df = pd.read_csv('data.csv')
x,y = df[['X1']].values,df['Y'].values

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w,x[i]) + b)
        cost += -y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)  
    cost /= m
    return cost 

def compute_gradient(x,y,w,b):
    m,n = x.shape 
    dj_dw,dj_db = np.zeros(n),0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i],w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i*x[i,j]
        dj_db += err_i
    dj_dw,dj_db = dj_dw/m,dj_db/m
    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,alpha,iters):
    w = copy.deepcopy(w_in)
    b = b_in
    j_hist,b1_val,b2_val = [],[],[]
    for i in range(iters + 1):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w -= alpha*dj_dw
        b1_val.append(w[0])
        b -= alpha*dj_db
        j_hist.append(compute_cost(x,y,w,b))
        print(f"Iterations {i} / {iters} b1:{w[0]} b0:{b} Cost: {compute_cost(x,y,w,b)}")
    return w,b,j_hist,b1_val,b2_val


w_tmp,b_tmp,alpha,iters = np.zeros(x.shape[1]),0.0,0.1,50
w,b,j_hist,b1_val,b2_val = gradient_descent(x,y,w_tmp,b_tmp,alpha,iters)
print(f"Updated parameters for slopes and intercept:\nb1,b2:{w}\nintercept:{b}\nMSE:{j_hist[-1]}")
        
def compute_accuracy(x,w,b):
    pred,num_correct = [],0
    m = x.shape[0]
    for i in range(m):
        y_pred = sigmoid(np.dot(x[i],w) + b)
        if y_pred > 0.5:
            pred.append(1)
            num_correct += 1
        elif y_pred < 0.5:
            pred.append(0)
            num_correct += 1
    accuracy = num_correct/len(pred)
    return accuracy,pred

accuracy,pred = compute_accuracy(x,w,b)
print(f"For {x} prediction is {pred} and Accuracy:{accuracy}")
plt.plot(b1_val,j_hist)
plt.title('B1 values vs errors')
plt.xlabel('B1 values')
plt.ylabel('Error values')
plt.show()

x_range = np.linspace(-2, 2, 400)
cost_y1 = [-np.log(1 - sigmoid(w[0] * xi + b)) for xi in x_range]
cost_y0 = [-np.log(sigmoid(w[0] * xi + b)) for xi in x_range]

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x_range, cost_y1, label='y=1')
plt.title('Binary Cross-Entropy for y=1')
plt.xlabel('X')
plt.ylabel('Cost')
plt.legend()

plt.subplot(122)
plt.plot(x_range, cost_y0, label='y=0')
plt.title('Binary Cross-Entropy for y=0')
plt.xlabel('X')
plt.ylabel('Cost')
plt.legend()

plt.tight_layout()
plt.show()
model = LogisticRegression(solver='sag')
model.fit(x,y)
y_pred = model.predict(x)
acc = accuracy_score(y,y_pred)
loss = log_loss(y,y_pred)
print(f"Sklearn model slope coeff:{model.coef_} and intercept:{model.intercept_}\nError:{loss}\nAccuracy:{acc}")


# 3. Use the above data set for two independent variable (X1,X2) and one dependent variable (Y) in CSV. Apply the Logistic Regression with SGD to perform the following.
# 
# a. Calculate the coefficients (B0, B1 and B2) and arrive at different values of B0, B1, B2, and error for 50 iterations of 5 epochs.
# 
# b. Apply the sigmoid function to get the prediction and calculate error.
# 
# c. From the predicted values calculate the accuracy.
# 
# d. Plot the graph of epoch (X-axis) v/s Accuracy (Y-axis).
# 
# f. Use scikit learn model to repeat the above steps and compare the results

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

df = pd.read_csv('data.csv')
x, y = df[['X1', 'X2']].values, df['Y'].values

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, x[i]) + b)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost /= m
    return cost

def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw, dj_db = np.zeros(n), 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * x[i, j]
        dj_db += err_i
    dj_dw, dj_db = dj_dw / m, dj_db / m
    return dj_dw, dj_db

def compute_accuracy(x, y, w, b):
    pred, num_correct = [], 0
    m = x.shape[0]
    for i in range(m):
        y_pred = sigmoid(np.dot(x[i], w) + b)
        if y_pred > 0.5:
            pred.append(1)
            num_correct += 1
        else:
            pred.append(0)
            #num_correct += 1
    accuracy = num_correct / len(pred)
    return accuracy

def stochastic_gradient_descent(x, y, w_in, b_in, alpha, iterations, epochs):
    w = copy.deepcopy(w_in)
    b = b_in
    j_hist, b1_val, b2_val, accuracy_list = [], [], [], []
    
    for iteration in range(iterations):
        for epoch in range(epochs):
            dj_dw, dj_db = compute_gradient(x, y, w, b)
            w -= alpha * dj_dw
            b -= alpha * dj_db
            j_hist.append(compute_cost(x, y, w, b))
            b1_val.append(w[0])
            b2_val.append(w[1])
            accuracy = compute_accuracy(x, y, w, b)
            accuracy_list.append(accuracy)
            print(f"Epoch {epoch}/{epochs} , Iteration {iteration}/{iterations}, Cost: {j_hist[-1]}, Accuracy: {accuracy}")
    
    return w, b, j_hist, b1_val, b2_val, accuracy_list

w_tmp, b_tmp, alpha, iterations, epochs = np.zeros(x.shape[1]), 0.0, 0.1, 50, 5
w, b, j_hist, b1_val, b2_val, accuracy_list = stochastic_gradient_descent(x, y, w_tmp, b_tmp, alpha, iterations, epochs)

print(f"Final coefficients: B0={b}, B1={w[0]}, B2={w[1]}")
print(f"Final error (MSE): {j_hist[-1]}")

# You can also plot the values of B0, B1, and B2 over iterations
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(range(iterations * epochs), b1_val)
plt.title('B1 values over iterations')
plt.xlabel('Iteration')
plt.ylabel('B1')

plt.subplot(132)
plt.plot(range(iterations * epochs), b2_val)
plt.title('B2 values over iterations')
plt.xlabel('Iteration')
plt.ylabel('B2')

plt.subplot(133)
plt.plot(range(iterations * epochs), j_hist)
plt.title('Error (MSE) over iterations')
plt.xlabel('Iteration')
plt.ylabel('Error')

plt.tight_layout()
plt.show()

# Plot the graph of epoch vs. accuracy
plt.figure(figsize=(8, 4))
plt.plot(range(iterations * epochs), accuracy_list)
plt.title('Epoch vs. Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

