#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# created by Shun-Lung Chang and Anna Ruby
# sh.chang@jacobs-university.de
# a.ruby@jacobs-university.de


import numpy as np
import matplotlib.pyplot as plt

class PCA():
    """
    Train the model using PCA
    """
    def fit(self, X, normalize=True, components=20):
        if components > X.shape[1] or components <=0:
            raise ValueError("Number of Components should be between [1, {0}]".format(X.shape[1]))
        if normalize:
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
        
        self.mean_X = np.mean(X, axis=0)
        self.centered_X = X - self.mean_X
        
        self.covariance_X = np.dot(self.centered_X.T, self.centered_X) / X.shape[0]
        
        # perform SVD on the covariance matrix
        self.components, self.singular_values, v = np.linalg.svd(self.covariance_X)
        
        # sort sigular values and components
        idx = np.argsort(self.singular_values)[::-1]
        self.singular_values = self.singular_values[idx]
        self.components = self.components[:, idx]
    
        self.reduced_components = self.components[:, 0:components]
        self.encoded_X = np.dot(self.centered_X, self.reduced_components)
        
        self.decoded_X = np.dot(self.encoded_X, self.reduced_components.T) + self.mean_X 
        

class LinearRegression():
    """
    Train the model using linear regression
    """
    def fit(self, X, y):
        # obtain the weights using pseudo-inverse matrix
        self.weights = np.linalg.pinv(X).dot(y)
        
    def predict(self, X):
        self.pred = X.dot(self.weights)
        
    def compute_error(self, y_vector, y_matrix):
        # compute mean square error
        self.MSE = np.mean(np.square(self.pred - y_matrix))
        
        # compute misclassification rate
        pred_class = np.argmax(self.pred, axis=1)
        self.MISS = sum(pred_class != y_vector) / len(pred_class)
        

if __name__ == "__main__":
    # load and divide data 
    X = np.loadtxt('input/mfeat-pix.txt')
    
    # split the dataset into training data and test data
    train_indices = [range(0 + 200*i, 100 + 200*i) for i in range(10)]
    test_indices = [range(100 + 200*i, 200 + 200*i) for i in range(10)]

    X_train = X[train_indices, ].reshape(1000, 240)
    X_test = X[test_indices, ].reshape(1000, 240)
    
    # extract encoded data from the first 20 components in a PCA model
    PCA_20 = PCA()
    PCA_20.fit(X_train, components=20)
    X_train_new = np.column_stack((np.ones(X_train.shape[0]), PCA_20.encoded_X))

    PCA_20.fit(X_test, components=20)
    X_test_new = np.column_stack((np.ones(X_test.shape[0]), PCA_20.encoded_X))
    
    # create class labels
    nb_classes = 10
    y_vector = np.array([i for i in range(10) for j in range(100)])
    y_matrix = np.eye(nb_classes)[y_vector]
    
    # fit a linear model with the training data
    lr = LinearRegression()
    lr.fit(X_train_new, y_matrix)
    lr.predict(X_train_new)
    lr.compute_error(y_vector, y_matrix)
    print(lr.MSE, lr.MISS)
    
    # fit a linear model with the test data
    lr.predict(X_test_new)
    lr.compute_error(y_vector, y_matrix)
    print(lr.MSE, lr.MISS)
    
    
    ## train regression models using different numbers of components
    # create empty vectors
    k_max = X.shape[1]
    train_MSE = np.zeros(k_max)
    train_MISS = np.zeros(k_max)
    test_MSE = np.zeros(k_max)
    test_MISS = np.zeros(k_max)
    
    # training models using different numbers of components
    for i in range(k_max):
        PCA_mod = PCA()
        PCA_mod.fit(X_train, components=i+1)
        X_train_new = np.column_stack((np.ones(X_train.shape[0]), PCA_mod.encoded_X))

        PCA_mod.fit(X_test, components=i+1)
        X_test_new = np.column_stack((np.ones(X_test.shape[0]), PCA_mod.encoded_X))

        lr = LinearRegression()
        lr.fit(X_train_new, y_matrix)
        lr.predict(X_train_new)
        lr.compute_error(y_vector, y_matrix)
        train_MSE[i] = lr.MSE
        train_MISS[i] = lr.MISS

        lr.predict(X_test_new)
        lr.compute_error(y_vector, y_matrix)
        test_MSE[i] = lr.MSE
        test_MISS[i] = lr.MISS
        
    # plot MSE and MISS
    plt.plot(train_MSE)
    plt.plot(train_MISS)
    plt.plot(test_MSE)
    plt.plot(test_MISS)

    plt.legend(['MSE (training)', 'MISS (training)', 'MSE (test)', 'MISS (test)'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    # obtain logarithmic MSE and MISS
    log_train_MSE = np.log(train_MSE)
    log_train_MISS = np.log(train_MISS)
    log_test_MSE = np.log(test_MSE)
    log_test_MISS = np.log(test_MISS)
    
    # plot logarithmic MSE and MISS
    plt.plot(log_train_MSE)
    plt.plot(log_train_MISS)
    plt.plot(log_test_MSE)
    plt.plot(log_test_MISS)

    plt.legend(['Log MSE (training)', 'Log MISS (training)', 'Log MSE (test)', 'Log MISS (test)'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    
    ## train regression models using 10 codebooks in a K-means model
    KMeans_10 = KMeans(n_clusters=10)
    KMeans_10.fit(X_train)
    X_train_new = np.column_stack((np.ones(X_train.shape[0]), KMeans_10.encoded_X))
    
    lr = LinearRegression()
    lr.fit(X_train_new, y_matrix)
    lr.predict(X_train_new)
    lr.compute_error(y_vector, y_matrix)
    print(lr.MSE, lr.MISS)
    
    KMeans_10.fit(X_test)
    X_test_new = np.column_stack((np.ones(X_test.shape[0]), KMeans_10.encoded_X))
    
    lr = LinearRegression()
    lr.fit(X_test_new, y_matrix)
    lr.predict(X_test_new)
    lr.compute_error(y_vector, y_matrix)
    print(lr.MSE, lr.MISS)
    
    
    ## train regression models using different number of codebooks
    # set maximum number of features and create empty vectors
    k_max = 300
    train_MSE = np.zeros(k_max)
    train_MISS = np.zeros(k_max)
    test_MSE = np.zeros(k_max)
    test_MISS = np.zeros(k_max)
    
    # training models using different numbers of codebooks
    for i in range(k_max):
        KMeans_mod = KMeans(n_clusters=i+1)
        KMeans_mod.fit(X_train)
        X_train_new = np.column_stack((np.ones(X_train.shape[0]), KMeans_mod.encoded_X))

        KMeans_mod.fit(X_test)
        X_test_new = np.column_stack((np.ones(X_test.shape[0]), KMeans_mod.encoded_X))

        lr = LinearRegression()
        lr.fit(X_train_new, y_matrix)
        lr.predict(X_train_new)
        lr.compute_error(y_vector, y_matrix)
        train_MSE[i] = lr.MSE
        train_MISS[i] = lr.MISS

        lr.predict(X_test_new)
        lr.compute_error(y_vector, y_matrix)
        test_MSE[i] = lr.MSE
        test_MISS[i] = lr.MISS
        
    # plot MSE and MISS
    plt.plot(train_MSE)
    plt.plot(train_MISS)
    plt.plot(test_MSE)
    plt.plot(test_MISS)

    plt.legend(['MSE (training)', 'MISS (training)', 'MSE (test)', 'MISS (test)'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    log_train_MSE = np.log(train_MSE)
    log_train_MISS = np.log(train_MISS)
    log_test_MSE = np.log(test_MSE)
    log_test_MISS = np.log(test_MISS)
    
    plt.plot(log_train_MSE)
    plt.plot(log_train_MISS)
    plt.plot(log_test_MSE)
    plt.plot(log_test_MISS)

    plt.legend(['Log MSE (training)', 'Log MISS (training)', 'Log MSE (test)', 'Log MISS (test)'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    