# importing the relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

transform = np.random.rand(5,5)

# squared error cost function
def cost(X,Y,theta):
    J = np.sum((X.dot(theta.T)-Y)**2)/(2*m)
    return J

# gradient descent algorithm
def gradient_descent(theta,X,Y,alpha,iterations=10000):
    costs = []
    for i in range(iterations):
        theta = theta - alpha*((X.T).dot(X.dot(theta.T)-Y)).T*(1/m)
        costs.append(cost(X,Y,theta))
        if(i%1000==0):
            print('Completed {} iterations. Cost = {}'.format(i,costs[i]))
    return theta.dot(transform)  

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

learning_rate = [0.000001,0.00001,0.0001,0.001]
rmse_list1 = []
rmse_list2 = []
rmse_list3 = []

print('Starting gradient descent using linear features ...')
for rate in learning_rate:
    print('learning rate = {}'.format(rate))
    theta = gradient_descent(initial_theta,X,Y,rate,10000)
    print('Theta obtained is {}'.format(theta))
    Y_pred = X_test.dot(theta.T)
    rmse_error = rmse(Y_test,Y_pred)
    rmse_list1.append(rmse_error)
    print('RMSE = {}'.format(rmse_error))

print('Starting gradient descent using quadratic features ...')
for rate in learning_rate:
    print('learning rate = {}'.format(rate))
    theta = gradient_descent(initial_theta,X*X,Y,rate,10000)
    print('Theta obtained is {}'.format(theta))
    Y_pred = (X_test**2).dot(theta.T)
    rmse_error = rmse(Y_test,Y_pred)
    rmse_list2.append(rmse_error)
    print('RMSE = {}'.format(rmse_error))

print('Starting gradient descent using cubic features ...')
for rate in learning_rate:
    print('learning rate = {}'.format(rate))
    theta = gradient_descent(initial_theta,X**3,Y,rate,40)
    print('Theta obtained is {}'.format(theta))
    Y_pred = (X_test**3).dot(theta.T)
    rmse_error = rmse(Y_test,Y_pred)
    rmse_list3.append(rmse_error)
    print('RMSE = {}'.format(rmse_error))
 
# plotting the rmse versus learning rate for different combination of features
f,axarr = plt.subplots(3,sharex=True)
axarr[0].plot(learning_rate,rmse_list1,'r',label='linear features')
axarr[1].plot(learning_rate,rmse_list2,'g',label='quadratic features')
axarr[2].plot(learning_rate,rmse_list3,label='cubic featues')
axarr[2].set(xlabel='learning rate',ylabel='rmse')
axarr[1].set(ylabel='rmse')
axarr[0].set(ylabel='rmse')
axarr[0].legend()
axarr[1].legend()
axarr[2].legend()
f.subplots_adjust(hspace=0.5)
f.suptitle('rmse vs learning rate for different feature combination')
plt.show()