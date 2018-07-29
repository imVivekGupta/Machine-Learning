# importing the required packages
import re
import string
import numpy as np
from random import shuffle
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from collections import Counter

# convert data to one-hot encoding format 
def one_hot_enc():
	for i in train:
		tokens = re.sub('['+string.punctuation+']',' ',i).split()		# use all punctuators as delimiters
		y = 1 if tokens[0]=='spam' else 0				# 0-ham 1-spam encoding used
		tokens = [ps.stem(token) for token in tokens[1:]]		# applying porter stemming
		v = []
		for word in vector:
			if word in tokens:
				v.append(1)
			else:
				v.append(0)
		train_ohe.append([v,y])
    
	for i in test:
		tokens = re.sub('['+string.punctuation+']',' ',i).split()
		y = 1 if tokens[0]=='spam' else 0
		tokens = [ps.stem(token) for token in tokens[1:]]
		v = []
		for word in vector:
			if word in tokens:
				v.append(1)
			else:
				v.append(0)
		test_ohe.append([v,y])    

# retrieve i-th training instance as |V|x1 vector and its label
def get_train_instance(i):
	t = train_ohe[i]
	x = (np.array([t[0]])).T 						# convert to |V|x1
	y = t[1]
	return x,y

# sigmoid function
def sigmoid(x):
	return (1/(1 + np.exp(-x)))

# activation function
def theta(s,activation_fn):
	return np.tanh(s) if activation_fn == 'tanh' else sigmoid(s)

# computes delta for final layer in backprop
def get_base_delta(h,y,activation_fn):
	# for squared error function
	delta = 2*(h-y)
	if activation_fn=='tanh':
		delta = delta*(1-h**2)
	elif activation_fn=='sigmoid':
		delta = delta*h*(1-h)
	else:									# softmax derivative
		temp = np.array([i*(1-i) for i in h])
		delta = delta*temp
	return delta

# cost function 
def squared_error(H,Y):
	J = np.sum((H-Y)**2)/(2*len(H))
	return J

# mean squared error in sample
def error_sample(sample,weights,activation_fn):
	h,y = [],[]
	all_sample = train_ohe if sample=='in' else test_ohe			# check if in-sample or out-sample
	for i in all_sample:
		y.append(i[1])
		v = i[0]
		x = (np.array([v])).T
		for w in weights:
			x = np.insert(x,0,1,0)
			s = (w.T).dot(x)
			x = theta(s,activation_fn)
		h.append(x[0][0])						# h is the output of the neural network
	return squared_error(np.array(h),np.array(y))    

# building a neural network
def backprop(activation_fn,eta,iterations = 1000):
	w1 = np.random.uniform(-1,0,(1+len(vector),100))/10		# random initialization of weights
	w2 = np.random.uniform(-1,0,(101,50))/10
	w3 = np.random.uniform(-1,0,(51,1))/10
	
	insample_error = list()
	outsample_error = list()
	for i in range(iterations):
		if i in error_checkpoints:					# compute error at defined check-points
			f.write('{} iterations: \n'.format(i))			# writing results in file
			print('{} iterations: '.format(i)) 
			insample_error.append(error_sample('in',[w1,w2,w3],activation_fn))
			outsample_error.append(error_sample('out',[w1,w2,w3],activation_fn))
			f.write('in_sample error = {} out_sample_error = {}\n'.format(insample_error[-1],outsample_error[-1]))
			print('in_sample error = {} out_sample_error = {}'.format(insample_error[-1],outsample_error[-1]))
		
		# forward prop
		x0,y = get_train_instance(np.random.randint(len(train)))	# obtain a random training instance
		x0 = np.insert(x0,0,1,0)					# add bias term
		s1 = (w1.T).dot(x0)						# s1 -> input to layer 1
		x1 = theta(s1,activation_fn)					# x1 -> output of layer 1
		x1 = np.insert(x1,0,1,0)					# add bias term
		s2 = (w2.T).dot(x1)						# s2 -> input to layer 2							
		x2 = theta(s2,activation_fn)					# x2 -> output of layer 2
		x2 = np.insert(x2,0,1,0)					# add bias term
		s3 = (w3.T).dot(x2)						# s3 -> input to layer 3
		h = theta(s3,activation_fn)					# h -> output of layer 3
		
		# backprop
		delta3 = get_base_delta(h,y,activation_fn)			# get delta of final layer
		if activation_fn=='tanh':					
		    delta2 = (1-x2**2)*(w3*delta3)				# compute delta2 from delta3
		    delta1 = (1-x1**2)*(w2.dot(delta2[1:]))  #101 x 1		# compute delta1 from delta2
		else:
		    delta2 = sigmoid(x2)*(1-sigmoid(x2))*(w3*delta3)		# compute delta2 from delta3
		    delta1 = sigmoid(x1)*(1-sigmoid(x1))*(w2.dot(delta2[1:])) # compute delta1 from delta2

		# update weights
		w1 = w1 - eta*(x0.dot((delta1[1:]).T))
		w2 = w2 - eta*(x1.dot((delta2[1:]).T))
		w3 = w3 - eta*(x2.dot((delta3).T))
	return [w1,w2,w3,insample_error,outsample_error]

# plots the error v/s iterations graphs
def plot(insample_error,outsample_error):
	x = error_checkpoints[:len(insample_error)]				# x is error checkpoints
	plt.plot(x,insample_error,label='in-sample')
	plt.plot(x,outsample_error,label='out-sample')
	plt.xlabel('Iterations')
	plt.ylabel('Mean squared error')
	plt.legend()
	plt.title('Error v/s iterations (Tanh A2)')
	plt.savefig('./result_files/plot_A2.png')

#computes  accuracy of model
def stats_summary(weights,activation_fn,threshold=0.5):
	tp,tn,fp,fn = 0,0,0,0
	for i in test_ohe:
		y = i[1]
		v = i[0]
		x = (np.array([v])).T
		for w in weights:
			x = np.insert(x,0,1,0)
			s = (w.T).dot(x)
			x = theta(s,activation_fn)
		h = 1 if x>=threshold else 0    
		if (h==1 and y==1):						# correctly classified spam (true positive)
			tp +=1
		elif (h==0 and y==0):						# correctly classified ham (true negative)
			tn +=1
		elif (h==0 and y==1):						# incorrectly classified actual spam (false negative)
			fn +=1
		else:
			fp +=1							# incorrectly classified actual ham (false positive)
	
	prec = tp/(tp+fp)							# computing precision
	recl = tp/(tp+fn)							# computing recall
	f1 = (2*prec*recl)/(prec+recl)						# computing F1 score
	acc = (tp+tn)/(tp+tn+fp+fn)						# computing accuracy

	f.write('\nSummary Statistics for threshold = {}\n'.format(threshold))
	f.write('Precision : {}\n'.format(prec))
	f.write('Recall : {}\n'.format(recl))
	f.write('F1 score : {}\n'.format(f1))
	f.write('Accuracy : {}\n'.format(acc))
	
	print('Precision : {}'.format(prec))        
	print('Recall : {}'.format(recl))        
	print('F1 score : {}'.format(f1))        
	print('Accuracy : {}'.format(acc))        

# reading data
with open('Assignment_2_data.txt','r') as f:
	print('reading data ...		',end='')
	lines = f.readlines()
	print('done.')        

# reading stopwords
with open('stopwords.txt','r') as f:
	stopwords = f.read().splitlines()

# convert to lower case
lines = [line.lower() for line in lines]

# store all tokens
print('finding all tokens ...')
all_tokens = []
for line in lines:
	tokens = re.sub('['+string.punctuation+']',' ',line).split()		# use all punctuations as delimiters
	all_tokens += (tokens[1:])						# removing 'ham'/'spam' tag

print('removing stopwords and stemming ...')
# filtered tokens after removing stopwords
filtered_tokens = [word for word in all_tokens if word not in stopwords]

# porter stemming
ps = PorterStemmer()
stemmed_tokens = [ps.stem(w) for w in filtered_tokens]

# again removing stopwords
stemmed_tokens = [word for word in stemmed_tokens if word not in stopwords]

# construct vector of 2000 most frequent tokens 
print('constructing word vector of 2000 most frequent words ...	',end='')
c = Counter(stemmed_tokens)
vector = [i[0] for i in c.most_common(2000)]
print('done.')

# segregating spams and hams for creating test and train sets
print('creating training and testing sets ... ',end='')
ham_instances,spam_instances = [],[]
for line in lines:
	tokens = re.sub('['+string.punctuation+']',' ',line).split()
	if tokens[0]=='ham':
		ham_instances.append(line)
	if tokens[0]=='spam':
		spam_instances.append(line)

# creating training and testing sets in 80-20 split  
shuffle(spam_instances)								# shuffling spam
shuffle(ham_instances)								# shuffling ham
m = int(0.8*len(spam_instances))
train = spam_instances[:m]								# 80% spam in training set
test = spam_instances[m:]								# 20% spam in testing set
m = int(0.8*len(ham_instances))
train += ham_instances[:m]								# 80% ham in training set
test += ham_instances[m:]								# 20% ham in testing set
shuffle(train)										# shuffling train set	
shuffle(test)										# shuffling test set
print('done.')

train_ohe,test_ohe = [],[]
# compute error at these number of iterations
error_checkpoints = [0,10,30,50,100,200,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000
			,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000
			,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000
			,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000
			,41000,42000,43000,44000,45000,46000,47000,48000,49000,49999]

print('converting data to one hot encoding ... 	',end='')
one_hot_enc()
print('done.')

# write results to file
f = open('./result_files/a2_results.txt','w')
f.write('	Activation function: tanh\n')
f.write('	Learning Rate (eta)	: 0.1\n')
f.write('	# Iterations: 50000\n')

print('starting backpropagation learning ...')
print('	Activation function: tanh')
print('	Learning Rate (eta)	: 0.1')
print('	# Iterations: 50000')

out = backprop('tanh',0.1,50000)				# run learning algorithm

print('Weights learned: ')
print('Layer 1: dimesions = {}'.format(out[0].shape))
print(out[0])
print('Layer 2: dimesions = {}'.format(out[1].shape))
print(out[1])
print('Layer 3: dimesions = {}'.format(out[2].shape))
print(out[2])

# setting spam classifier threshold based on spam distribution in data
threshold = len(spam_instances)/len(lines)
print('computing testing statistics (threshold={})...		'.format(threshold))
stats_summary(out[:3],'tanh',threshold)

# setting threshold to 0.5 for spam mail classifier
print('computing testing statistics (threshold=0.5)...		')
stats_summary(out[:3],'tanh')

print('generating plot ...	 ',end='')
plot(out[3],out[4])
print('done.')

# writing results to file
np.set_printoptions(threshold=np.nan)
f.write('\nWeights learned: \n')
f.write('Layer 1: dimesions = {}\n'.format(out[0].shape))
f.write(str(out[0]))
f.write('Layer 2: dimesions = {}\n'.format(out[1].shape))
f.write(str(out[1]))
f.write('Layer 3: dimesions = {}\n'.format(out[2].shape))
f.write(str(out[2]))
f.close()