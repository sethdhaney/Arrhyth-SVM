from random import *
from csv import reader
from pylab import *
from sklearn import svm
from sklearn.neural_network import MLPClassifier

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo







# Calculate and plot learning curve
def learningCurve(X,y,X_val,y_val,C_opt):
	m = shape( X )[0]
	error_train = list()
	error_val = list()

	m_val = shape( X_val )[0]

	idx1 = find(y==1)[0]
	idx0 = find(y==2)[0]
	for i in range( 0, m ):
		if (i%5==0):
			SVM_CL = svm.SVC(C=C_opt, kernel='linear')
			#Make sure training set has positive and negative examples
			X_tst = np.concatenate((X[[idx0,idx1], :], X[0:i+1,:]),axis=0)
			y_tst = np.concatenate((y[[idx0,idx1]], y[0:i+1]))
	
			SVM_CL.fit(X_tst,y_tst)
	
			pr = SVM_CL.predict(X_tst)
			error_train.append(len(find(pr != y_tst))/len(y_tst))
			pr = SVM_CL.predict(X_val)
			error_val.append(len(find(pr != y_val))/len(y_val))

	error_train = array(error_train)
	error_val   = array(error_val)

	# number of training examples
	plt.ylabel('Error')
	plt.xlabel('Number of training examples')
	plt.plot( error_train, color='b', linewidth=2, label='Train' )
	plt.plot( error_val, color='g', linewidth=2, label='Cross Validation' )
	plt.legend()
	plt.show()

	return error_train, error_val



def validationCurve( X_Tr, Y_Tr, X_val, y_val, C_vec ):

	error_train = []
	error_val = []

	for C_tst in C_vec:
		SVM_CL = svm.SVC(C=C_tst, kernel='linear')
		SVM_CL.fit(X_Tr,Y_Tr)

		#Predict
		pr = SVM_CL.predict(X_Tr)
		error_train.append(len(find(pr != Y_Tr))/len(Y_Tr))
		pr = SVM_CL.predict(X_val)
		error_val.append(len(find(pr != y_val))/len(y_val))
		

	error_train = array( error_train )
	error_val = array( error_val )

	plt.ylabel('Error')
	plt.xlabel('Lambda')
	plt.plot( C_vec, error_train, 'b', label='Train' )
	plt.plot( C_vec, error_val, 'g', label='Cross Validation' )
	plt.xscale( 'log' )
	plt.legend()
	plt.show()

	return error_train, error_val


#Create polynomial (degree p+1) features
def polyFeatures(X,p):

    X_poly = copy(X)


    for i in range(1,p):                                                     
        X_poly = c_[X_poly,X**(i+1)]                                         
                                                                             
                                                                             
    return X_poly                                                            
                                                                             
def featureNormalize(X_poly):                                                
                                                                             
    mu = mean(X_poly,axis=0)                                                 
    diff = X_poly - mu                                                       
                                                                             
    sigma = std(diff,axis=0,ddof =1)
    #Handle cases when sigma=0
    idx = find(sigma==0)
    sigma[idx]=1
    norm = diff/sigma                                                        
                                                                             
    return norm, mu, sigma                                                   
                                                                             





#########################

seed(1)

# LOAD CSV File
filename = 'arrhythmia.data'

data = np.genfromtxt(filename, delimiter=",")
dataT = np.transpose(data)
Y = np.transpose(dataT[-1])
X = np.transpose(dataT[0:-2])

NF = len(X[0])
NE = len(X)

#########################
# DEAL WITH MISSING DATA
# Find missing data

# 92% of missing data is due to field 13. I remove this from the
# dataset and then remove the remaing 32 records with other missing
# records. This process retains 

X_orig = copy(X)
Y_orig = copy(Y)

idx = list(range(0,NF))
idx.remove(13)
NF = NF-1

X_d13 = X[:,idx]
i=0
X_del = list()
Y_del = list()
idxMissing = list()
for row in X_d13:
	bMissing = False
	for j in range(0,NF):
		if(np.isnan(row[j])):
			idxMissing.append([i,j]) 	
			bMissing = True
	
	if (bMissing==False):
		X_del.append(row)
		Y_del.append(Y[i])
	
	i = i+1

X = array(X_del)
Y = array(Y_del)
X, mu, sigma = featureNormalize(X)
NF = X.shape[1]

#################



# Split Training and Validation Set
p_Tr = 0.8
rnds = rand(len(Y))
iTr = find(rnds<p_Tr)
iVal = find(rnds>=p_Tr)
X_Tr = X[iTr,:]
Y_Tr = Y[iTr]
X_Val = X[iVal,:]
Y_Val = Y[iVal]


#Regularization
C_opt = 1.0


#Validation Curve (Optimizing for Regularization)
reg_vec = array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]).T
figure(1)
clf()
C_err_tr, C_err_val = validationCurve(X_Tr, Y_Tr, X_Val, Y_Val, reg_vec)
C_opt = reg_vec[argmax(-C_err_val)]
print('Optimal C = ', C_opt)


#Create Learning curve
print('Running Learning Curve...')
figure(2)
clf()
m_err_tr, m_err_val = learningCurve(X_Tr,Y_Tr,X_Val,Y_Val,C_opt)





#Train
SVM_CL = svm.SVC(C=C_opt, kernel='linear')
SVM_CL.fit(X_Tr,Y_Tr)

#Predict on Training Set
pr_Tr = SVM_CL.predict(X_Tr)

err_Tr = len(find(pr_Tr != Y_Tr))/len(Y_Tr)
print('Error on Training Set', err_Tr)

#Predict on Validation Set
pr_Val = SVM_CL.predict(X_Val)

err_Val = len(find(pr_Val != Y_Val))/len(Y_Val)
print('Error on Test Set', err_Val)





