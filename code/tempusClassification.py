import numpy as np
import sys
import pdb
import time


feature_vector_size = 16562
num_of_training_samples = 420
num_of_test_samples = 110
num_of_iterations=500
neeta = 1
input_data_url = "C:\\Users\\faizi\\Downloads\\DScasestudy1\\takehome1.txt"

train_data = np.zeros(shape=(num_of_training_samples, feature_vector_size))
train_data_output = np.zeros(shape=(num_of_training_samples, 1))
test_data = np.zeros(shape=(num_of_test_samples, feature_vector_size))
test_data_output = np.zeros(shape=(num_of_test_samples, 1))




# sigmoid function
# this function makes sure that the value of x is between
# for normalization purpose & also the function is continuous
def sigmoid(x):
    return 1/(1+np.exp(-x))

# calculates the output values given input and weights
def feed_forward(data, weights):
    U = np.dot(data/len(data[0]),weights)
    Y = sigmoid(U)
    return Y

# calculate weights
def find_weights(train_data, train_data_output, neeta = 0.01):
    np.random.seed(1)
    # initialize weights randomly between 0 & 1 and normalize them
    weights = 2*np.random.random((feature_vector_size,1)) - 1

    for i in xrange(num_of_iterations):
        Y = feed_forward(train_data, weights)
        # Back propagation using gradient descent for
        # the Error function: E(w) = (1/2)*((Y-Output)^2)
        # which is also called mean squared error
        temp1 = np.multiply(Y,(1-Y))
        temp2 = np.multiply(temp1, Y-train_data_output)
        gradient = np.dot(np.transpose(train_data),temp2)
        # weights_new = weights_old - neeta*input*Y*(1-Y)*(Y-Output)
        weights = weights - np.multiply(neeta, gradient)

    return weights


def accuracy(weights, test_data, test_data_output):
    y = feed_forward(test_data,weights)
    error_count=0
    for i in xrange(len(test_data)):
        if(test_data_output[i] == 0 and y[i]>=0.5 ):
            error_count +=1
        if(test_data_output[i] == 1 and y[i]<0.5 ):
            error_count +=1

    acc = float(len(test_data)-error_count)/float(len(test_data))
    print("error_count = "+str(error_count)+", test_data_length = "+str(len(test_data)))
    #pdb.set_trace()
    return acc



print("Taking Input ...")
with open(input_data_url) as f:
    f.readline()
    for i in range(0,num_of_training_samples):
        line = f.readline()
        data = line.split()
        train_data_output[i]=int(data[0])
        for j in range(0,feature_vector_size):
            train_data[i,j] = int(data[j+1])
    for i in range(0,num_of_test_samples):
        line = f.readline()
        data = line.split()
        test_data_output[i]=int(data[0])
        for j in range(0,feature_vector_size):
            test_data[i,j] = int(data[j+1])

print("finding weights ...")
start = time.time()
w = find_weights(train_data, train_data_output, neeta=neeta)
end = time.time()
print("time to calculate weights is = "+str(end-start)+" seconds")
print("calculating accuracy for the test data ...")
a = accuracy(w, test_data, test_data_output)
print("a = "+str(a))




