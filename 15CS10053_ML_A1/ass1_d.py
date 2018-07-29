# importing the relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

transform = np.random.rand(5,5)

# mean squared error cost function
def cost_mean_squared(X,Y,theta):
    J = np.sum((X.dot(theta.T)-Y)**2)/(2*m)
    return J

# mean absolute error cost function
def cost_mean_absolute(X,Y,theta):
    J = np.sum(abs(X.dot(theta.T)-Y))/(m)
    return J

# mean cubed error cost function
def cost_mean_cubed(X,Y,theta):
    J = np.sum((X.dot(theta.T)-Y)**3)/(3*m)
    return J

# gradient descent algorithm for mean squared error cost function
def gradient_descent_mean_squared(theta,X,Y,alpha,iterations=10000):
    costs = []
    for i in range(iterations):
        theta = theta - alpha*((X.T).dot(X.dot(theta.T)-Y)).T*(1/m)
        costs.append(cost_mean_squared(X,Y,theta))
        if(i%100==0):
            print('Completed {} iterations. Cost = {}'.format(i,costs[i]))
    theta = (theta.dot(transform))*(df1['price'].std())
    theta[0] = theta[0]+df1['price'].mean()
    return  theta

# gradient descent algorithm for mean absolute error cost function
def gradient_descent_mean_absolute(theta,X,Y,alpha,iterations=10000):
    costs = []
    for i in range(iterations):
        theta = theta - alpha*(1/m)*((np.sign(X.dot(theta.T)-Y)).T).dot(X)
        costs.append(cost_mean_absolute(X,Y,theta))
        if(i%100==0):
            print('Completed {} iterations. Cost = {}'.format(i,costs[i]))
    theta = (theta.dot(transform))*(df1['price'].std())
    theta[0] = theta[0]+df1['price'].mean()
    return  theta

# gradient descent algorithm for mean cubed error cost function
def gradient_descent_mean_cubed(theta,X,Y,alpha,iterations=10000):
    costs = []
    for i in range(iterations):
        theta = theta - alpha*(1/m)*(((X.dot(theta.T)-Y)**2).T).dot(X)
        costs.append(cost_mean_cubed(X,Y,theta))
        if(i%10==0):
            print('Completed {} iterations. Cost = {}'.format(i,costs[i]))
    theta = (theta.dot(transform))*(df1['price'].std())
    theta[0] = theta[0]+df1['price'].mean()
    return  theta

# root mean square error metric
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

print('reading data ...')
df = pd.read_csv('kc_house_data.csv',encoding = 'utf-8')
df1 = df.copy()
print('normalising data ...')
df['sqft'] = (df['sqft']-df['sqft'].mean())/df['sqft'].std()
df['floors'] = (df['floors']-df['floors'].mean())/df['floors'].std()
df['bedrooms'] = (df['bedrooms']-df['bedrooms'].mean())/df['bedrooms'].std()
df['bathrooms'] = (df['bathrooms']-df['bathrooms'].mean())/df['bathrooms'].std()
df['price'] = (df['price']-df['price'].mean())/df['price'].std()

# for converting theta back to denormalised form
transform = np.array([[1,0,0,0,0],[ -df1['sqft'].mean()/df1['sqft'].std() , 1/df1['sqft'].std(), 0, 0, 0], [-df1['floors'].mean()/df1['floors'].std(), 0, 1/df1['floors'].std(), 0, 0] ,[-df1['bedrooms'].mean()/df1['bedrooms'].std(), 0, 0, 1/df1['bedrooms'].std(), 0] ,[-df1['bathrooms'].mean()/df1['bathrooms'].std(), 0, 0, 0, 1/df1['bathrooms'].std(),]])
m = int(len(df)*.8)
print('{} records in dataset'.format(m))

print('splitting into training and testing sets')
train = df[:m]
test = df1[m:]

alpha = 0.005
initial_theta = np.random.random(5)
x0 = np.ones(m)
X = np.array([x0,train['sqft'],train['floors'],train['bedrooms'],train['bathrooms']]).T
Y = np.array(train['price'])

n = len(df)-m
x0_test = np.ones(n)
X_test = np.array([x0_test,test['sqft'],test['floors'],test['bedrooms'],test['bathrooms']]).T
Y_test = np.array(test['price'])

learning_rate = [0.001,0.003,0.009]
rmse_list1 = []
rmse_list2 = []
rmse_list3 = []

print('Starting gradient descent using mean absolute error Cost function ...')
for rate in learning_rate:
    print('learning rate = {}'.format(rate))
    theta = gradient_descent_mean_absolute(initial_theta,X,Y,rate,1000)
    print('Theta obtained is {}'.format(theta))
    Y_pred = X_test.dot(theta.T)
    rmse_error = rmse(Y_test,Y_pred)
    rmse_list1.append(rmse_error)
    print('RMSE = {}'.format(rmse_error))

print('Starting gradient descent using mean squared error Cost function ...')
for rate in learning_rate:
    print('learning rate = {}'.format(rate))
    theta = gradient_descent_mean_squared(initial_theta,X,Y,rate,1000)
    print('Theta obtained is {}'.format(theta))
    Y_pred = X_test.dot(theta.T)
    rmse_error = rmse(Y_test,Y_pred)
    rmse_list2.append(rmse_error)
    print('RMSE = {}'.format(rmse_error))

print('Starting gradient descent using mean cubed error Cost function ...')
for rate in learning_rate:
    print('learning rate = {}'.format(rate))
    theta = gradient_descent_mean_cubed(initial_theta,X,Y,rate,110)
    print('Theta obtained is {}'.format(theta))
    Y_pred = X_test.dot(theta.T)
    rmse_error = rmse(Y_test,Y_pred)
    rmse_list3.append(rmse_error)
    print('RMSE = {}'.format(rmse_error))

# plotting the rmse versus learning rate for different cost functions
f,axarr = plt.subplots(3,sharex=True)
axarr[0].plot(learning_rate,rmse_list1,'r',label='absolute error')
axarr[1].plot(learning_rate,rmse_list2,'g',label='squared error')
axarr[2].plot(learning_rate,rmse_list3,label='cubed error')
axarr[2].set(xlabel='learning rate',ylabel='rmse')
axarr[1].set(ylabel='rmse')
axarr[0].set(ylabel='rmse')
axarr[0].legend()
axarr[1].legend()
axarr[2].legend()
f.subplots_adjust(hspace=0.5)
f.suptitle('rmse vs learning rate for different cost functions')
plt.show()    