"""
Adam Schaffroth
CPSC 392 Prof. Linstead
HW3 KNN
ID: 2253832


At least this one runs...

"""
import math
import pandas as pd
import numpy as np
import seaborn as sns



NUM_ATTRIBUTES = 4


train_df = pd.read_csv("C:/Users/adams/OneDrive/Documents/CPSC 392/Assignment 3/sample_train.csv", header = None, names = ["petal_length","petal_width","sepal_length","sepal_width","iris_type"])
test_df = pd.read_csv("C:/Users/adams/OneDrive/Documents/CPSC 392/Assignment 3/sample_test.csv", header = None, names = ["petal_length","petal_width","sepal_length","sepal_width","iris_type"])



#TIME FOR EXPLORATORY DATA ANALYSIS
train_df.iloc[:,0:4] = train_df.iloc[:,0:4].apply(lambda x: x.fillna(x.mean()), axis=0)#replace NAs w/ col mean
train_df
train_df.iloc[:,0:4] =  train_df.iloc[:,0:4].apply(lambda x: x.where(np.abs(x-x.mean()) <= 3*x.std(), other = x.mean()), axis=0)
#inplace = True arg was not working...
"""
sns.boxplot(x = train_df.sepal_length)
sns.boxplot(x = train_df.sepal_width)
sns.boxplot(x = train_df.petal_length)
sns.boxplot(x = train_df.petal_width)
"""

#train_df.to_csv("C:/Users/adams/OneDrive/Documents/CPSC 392/Assignment 3/")
TRAIN_DATA_FILE = "C:/Users/adams/OneDrive/Documents/CPSC 392/Assignment 3/training_data_manual.csv"
TEST_DATA_FILE = "C:/Users/adams/OneDrive/Documents/CPSC 392/Assignment 3/sample_test.csv"


#read the train file and return the data matrix and the target variable to predict
def readData(fname):
	data = []
	labels = []
	f = open(fname,"r")
	for i in f:
		instance = i.split(",")
		vector = []
		for j in range(NUM_ATTRIBUTES):
			vector.append(float(instance[j]))
		data.append(vector)
		labels.append(instance[NUM_ATTRIBUTES])
	f.close()
	return [data,labels]

#compute the dot product of vectors represented as lists
def dotProduct(vecA,vecB):
	sum = 0.0
	for i in range(NUM_ATTRIBUTES):
		sum += vecA[i]*vecB[i]
	return sum

#compute the cosine similarity of 2 vectors represented as lists
def cosDistance(vecA,vecB):
	normA = math.sqrt(dotProduct(vecA,vecA))
	normB = math.sqrt(dotProduct(vecB,vecB))
	return dotProduct(vecA,vecB)/(normA*normB) 
	
#compare predicted labels to truth labels. Identify errors and print accuracy
def printAccuracy(pred,truth):
	total = 0.0
	correct= 0.0
	for i in range(len(pred)):
		total += 1.0
		if pred[i]==truth[i]:
			correct += 1.0
		else:
			print("Predicted that test point ",i," was ",pred[i], "but it is actually ",truth[i])
	print("The accuracy is: ", 100*(correct/total), " percent")
	
#The KNN algorithm. Predicts the label for each test data set instance and adds to a list. Returns the list as output
def knn(train_data,train_labels,test_data):
    predictions = []
    #implement KNN here
	#for each test data point predict the label and add your prediction to the preditions list
	#compare to every data point in train_data using cosDistance by making a call to the above function
	#find the index, c, of the closest data point
    for x in range(len(test_data)):
        closestDistance = -100.0#initialize to something less than -1; BUT HIGHER NUMBERS ARE CLOSER (max 1)!!!!!
        tempDistance = -1.0#initialize everything outside , must be a the lowest reasonable distance; floating pt
        index_of_closest = -1#initialize to final ele
        for i in range(len(train_data)):
            tempDistance = cosDistance(test_data[x], train_data[i])
            if closestDistance < tempDistance:
                closestDistance = tempDistance
                index_of_closest = i
        predictions.append(train_labels[index_of_closest])
    return predictions


#this is the main routine of the program. You should not have to modify anything here
if __name__ == "__main__":
	train_matrix = readData(TRAIN_DATA_FILE)
	train_data = train_matrix[0]
	train_labels = train_matrix[1]
	test_matrix = readData(TEST_DATA_FILE)
	test_data = test_matrix[0]
	test_labels = test_matrix[1]
	predictions = knn(train_data,train_labels,test_data)
	printAccuracy(predictions,test_labels)
