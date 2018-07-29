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
def gradient_descent(theta,X,Y,X_test,Y_test,alpha,iterations=10000):
    rmse_list = []
    for i in range(iterations):
        theta = theta - alpha*((X.T).dot(X.dot(theta.T)-Y)).T*(1/m)
        theta1 = theta.dot(transform)
        rmse_list.append(rmse(Y_test,X_test.dot(theta1.T)))
        if(i%100==0):
            print('Completed {} iterations. RMSE = {}'.format(i,rmse_list[i]))       	
    return (theta1,rmse_list)  

# iterative reweighted least square algorithm
def irls(theta,X,Y,X_test,Y_test,iterations=10000):
    rmse_list = []
    for i in range(iterations):
        theta = theta - (np.linalg.inv((X.T).dot(X)).T).dot((X.T).dot(X.dot(theta.T)-Y))
        theta1 = theta.dot(transform)
        rmse_list.append(rmse(Y_test,X_test.dot(theta1.T)))
        if(i%100==0):
            print('Completed {} iterations. RMSE = {}'.format(i,rmse_list[i]))
    return (theta1,rmse_list)  

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

print('Starting gradient descent without regularization ...')
theta,rmse1 = gradient_descent(initial_theta,X,Y,X_test,Y_test,alpha,1000)
print('Theta obtained is {}'.format(theta))

Y_pred = X_test.dot(theta.T)
print('rmse without regularization for a linear regression model is {}'.format(rmse(Y_test,Y_pred)))

print('Starting iterative re-weighted least square ...')
theta,rmse2 = irls(initial_theta,X,Y,X_test,Y_test,1000)
print('Theta obtained for IRLS is {}'.format(theta))
Y_pred = X_test.dot(theta.T)
print('rmse with iterative re-weighted least square model is {}'.format(rmse(Y_test,Y_pred)))

# plotting the rmse versus number of iterations
plt.plot([i for i in range(1000)],rmse1,label='grad-descent')
plt.plot([i for i in range(1000)],rmse2,'--',label='irls')
plt.legend()
plt.title('rmse v/s iterations')
plt.xlabel('no of iterations')
plt.ylabel('rmse')
plt.show()