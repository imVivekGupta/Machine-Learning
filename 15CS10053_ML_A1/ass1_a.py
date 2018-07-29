# importing the relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        if(i%100==1):
            print('Completed {} iterations. Cost = {}'.format(i,costs[i]))
            if(costs[i-1]-costs[i] < 0.01):		# declare convergence if cost does not change by more than 0.01
                print('Convergence achieved , breaking ...') 	
                break       	
    return (theta,costs)  

# regularised gradient descent algorithm
def gradient_descent_regularised(theta,lambd,X,Y,alpha,iterations=100):
    costs = []
    t = (1-(alpha*lambd/m))
    x = np.array([1,t,t,t,t])
    for i in range(iterations):
        theta = theta*x - alpha*((X.T).dot(X.dot(theta.T)-Y)).T*(1/m)
        costs.append(cost(X,Y,theta))
    return theta,costs

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
theta = np.random.random(5)
x0 = np.ones(m)
X = np.array([x0,train['sqft'],train['floors'],train['bedrooms'],train['bathrooms']]).T
Y = np.array(train['price'])

n = len(df)-m
x0_test = np.ones(n)
X_test = np.array([x0_test,test['sqft'],test['floors'],test['bedrooms'],test['bathrooms']]).T
Y_test = np.array(test['price'])

print('Starting gradient descent without regularization ...')
theta,costs = gradient_descent(theta,X,Y,alpha)
theta = theta.dot(transform)
print('Value of model parameters: {}'.format(theta))

Y_pred = X_test.dot(theta.T)
print('rmse without regularization for a linear regression model is {}'.format(rmse(Y_test,Y_pred)))

print('For regularization:')
rmse_list = []
costs = []
lambdas = [i*0.1 for i in range(0,100,3)]
for lambd in lambdas:
    theta = np.random.random(5)
    theta,costs = gradient_descent_regularised(theta,lambd,X,Y,alpha,iterations=1000)
    theta = theta.dot(transform)
    Y_pred = X_test.dot(theta.T)
    rmse_error = rmse(Y_test,Y_pred)
    rmse_list.append(rmse_error)
    print('lambda = {} , rmse = {}, theta = {}'.format(lambd,rmse_error,theta))
    
# plotting the rmse versus regularization parameter
plt.plot(lambdas,rmse_list)
plt.title('rmse v/s regularization')
plt.xlabel('lambda')
plt.ylabel('rmse')
plt.show()  