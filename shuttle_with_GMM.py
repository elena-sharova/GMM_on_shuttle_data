# -*- coding: utf-8 -*-
"""
Created on Fri Sep 02 14:20:16 2016

@author: Elena Sharova
"""
import pandas
from sklearn import mixture
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


# load train data from downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup99.html
train_file_name="C:\\Users\\Elena\\Desktop\\Codefying\\GMM\\shuttle.train"
test_file_name="C:\\Users\\Elena\\Desktop\\Codefying\\GMM\\shuttle.test"

train_data_orig = pandas.DataFrame(pandas.read_csv(filepath_or_buffer=train_file_name, sep=' ', header=0,names=['p1','p2','p3', 'p4','p5','p6', 'p7','p8','p9','class']))
test_data_orig = pandas.DataFrame(pandas.read_csv(filepath_or_buffer=test_file_name, sep=' ', header=0,names=['p1','p2','p3', 'p4','p5','p6', 'p7','p8','p9','class']))
# Shuffle the train data
train_data_orig = train_data_orig.sample(frac=1).reset_index(drop=True)
# Reduce the number of columns to just p2 and p5 and normalize (can be optional)
cols=['p2','p5']

classes = [1,3,4,5,6,7]
results = []
for c in classes:

    train_data = train_data_orig.loc[(train_data_orig['class']==c) | (train_data_orig['class']==2)]

    #train_data_norm=pandas.DataFrame(data=normalize(train_data[cols]), columns=cols)  # optional normalise
    train_data_norm=train_data[cols]
    
    test_data = test_data_orig.loc[(test_data_orig['class']==c) | (test_data_orig['class']==2)]
    #test_data_norm=pandas.DataFrame(data=normalize(test_data[cols]), columns=cols)
    test_data_norm=test_data[cols]
    test_data_reduced=test_data[['p2','p5', 'class']]

    # Train the model 
    gm = mixture.GMM(n_components=2, covariance_type='diag', n_iter=9100, tol=0.0001, n_init=10, random_state=1, verbose=False)
    gm.fit(train_data_norm) 
    
    # Generate predictions on the test data
    predictions=gm.predict(test_data_norm)

    N = len(predictions)
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    # when checking for class 2 we should take into account which is the dominant class in the test set (the less dominant gets assigned 1)
    if len(test_data_reduced[test_data_reduced['class']==2]) > len(test_data_reduced[test_data_reduced['class']==c]):
        for pred, test in zip(predictions, test_data_reduced['class']):
            if pred==0 and test==2:
                tp+=1
            elif pred==1 and test==c:
                tn+=1
            elif pred==0 and test==c:
                fp+=1
            elif pred==1 and test==2:
                fn+=1
    else:
        for pred, test in zip(predictions, test_data_reduced['class']):
            if pred==1 and test==2:
                tp+=1
            elif pred==0 and test==c:
                tn+=1
            elif pred==1 and test==c:
                fp+=1
            elif pred==0 and test==2:
                fn+=1
    print "Testing against class %s, correctly predicted %s of class 2" %(c, tp)
    results.append([tp/(tp+fn),tn/(tn+fp),fp/(fp+tn),fn/(tp+fn)])  # there are 13 records with class 2
    # Plot the test data distribution for p2 and p1
    if (c==1):
        test_data_reduced_normal=test_data_reduced[test_data_reduced['class']!=2]
        test_data_reduced_anomaly=test_data_reduced[test_data_reduced['class']==2]
        plt.subplot(311)
        plt.scatter(test_data_reduced_normal['p2'].tolist(),test_data_reduced_normal['class'].tolist(), s=100, marker='^', c='g', alpha=0.75, label="normal")
        plt.scatter(test_data_reduced_anomaly['p2'].tolist(),test_data_reduced_anomaly['class'].tolist(), s=100, marker='^', c='r', alpha=0.75, label="anomaly")
        plt.grid(True)
        plt.legend(loc=2)
        plt.xlabel("p2")
        plt.ylabel("class")
        plt.subplot(312)
        plt.scatter(test_data_reduced_normal['p5'].tolist(),test_data_reduced_normal['class'].tolist(), s=100, marker='o', alpha=0.75, label="normal",c='g')
        plt.scatter(test_data_reduced_anomaly['p5'].tolist(),test_data_reduced_anomaly['class'].tolist(), s=100,marker='o', c='r', alpha=0.75, label="anomaly")
        plt.grid(True)
        plt.xlabel("p5")
        plt.ylabel("class")
        plt.legend(loc=1)

plt.subplot(313)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_visible(False)
cur_axes.axes.get_yaxis().set_visible(False)
plt.table(cellText=results,rowLabels=("class 1&2", "class 3&2", "class 4&2", "class 5&2", "class 6&2", "class 7&2"),
      colLabels=("TP","TN", "FP", "FN"),loc='center')
      
plt.show()