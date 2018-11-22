import numpy as np
import pylab as pl
import math
import time
import cvxopt
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
from scipy.optimize import fmin
from numpy import *


class ReadData:
    """
    Class which
    reads the given dataset

    """

    def read_data(self, dataset):
        """
        Function which reads the
        given dataset

        Parameter
        -------------------------------
        dataset : The dataset to be read

        Returns
        --------------------------------
        X : Features
        Y : Labels
        """

        data = np.loadtxt(dataset)
        X = data[:, 0:2]
        Y = data[:, 2:3]
        Y = Y.reshape(len(Y),)

        return X, Y



class SVM:
    '''
    Class which implements
    primal and dual form of SVM
    and all the three kernels - linear,
    polynomial and guassian kernel.
    Required for Q2 and Q3

    '''

    def plotDecisionBoundarySVM(self, X, Y, predictSVM, values, title = ""):
        '''
        Function which plots the
        SVM graphs

        '''
        # Plot the decision boundary. For that, we will asign a score to
        # each point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = max((x_max-x_min)/200., (y_max-y_min)/200.)
        xx, yy = meshgrid(arange(x_min, x_max, h),
                          arange(y_min, y_max, h))
        V = np.array([[x1, x2] for x1, x2 in zip(np.ravel(xx), np.ravel(yy))])
        zz = predictSVM(V).reshape(xx.shape)
        pl.figure()
        CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
        pl.clabel(CS, fontsize=9, inline=1)
        # Plot the training points
        pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
        pl.title(title)
        pl.axis('tight')
        pl.show(block=False)
        time.sleep(4)
        pl.close()

    def primal_fit(self, X, Y, C):
        '''
        Implements primal form of SVM,
        the primal form of SVM is solved
        using cvxopt and alphas are calculated

        Parameters
        ----------------------------------------
        X : Features
        Y : Labels
        C = Regularization Parameter

        Returns
        ------------------------------------------
        W    : The weight vector
        bias : The bias term

        '''

        # Calculating length and number of features
        length, number_features = X.shape

        # Creating P
        size = length + number_features + 1

        P = np.zeros((size, size))

        for i in range(number_features):
            P[i,i] = 1

        # Slack variable term
        c = np.vstack([np.zeros((number_features+1,1)),
        C*np.ones((length,1))])

        # Setting up of ...
        A = np.zeros((2*length, number_features+1+length))

        A[:length, 0:number_features] = Y[:, None] * X

        A[:length, number_features] = Y.T
        A[:length, number_features+1:]  = np.eye(length)
        A[length:, number_features+1:]  = np.eye(length)

        A = -A

        G = np.zeros((2*length, 1))

        G[:length] = -1

        svm_solution = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(c), cvxopt.matrix(A), cvxopt.matrix(G))

        alphas = np.array(svm_solution['x'])

        w = np.array(svm_solution['x'][:number_features])

        bias = np.array(svm_solution['x'][number_features])

        w = w.reshape(2,)

        return w, bias

    def count_mistakes(self, Y, Y_pred):
        '''
        Function which counts
        the number of mistakes
        in the given data

        '''
        return np.sum(Y_pred != Y)

    def non_kernel_predict(self, X, W, b):
        '''
        Implements predict function for non kernel
        primal case of SVM
        '''
        pred_y = np.dot(W,X.T ) + b

        # Uncomment this to print predictions
        #print("pred", np.sign(pred_y))

        return np.sign(pred_y)

    def linear_kernel(self, X, Y, parameter):
        '''
        Linear kernel
        '''
        result = np.dot(X.T, Y)
        return result

    def polynomial_kernel(self, X, y, dim):
        '''
        Polynomial kernel
        '''

        result = (np.dot(X.T, y) + 1)**dim
        return result

    def guassian_kernel(self, X, Y, sigma):
        '''
        Guassian kernel
        takes sigma as parameter
        '''
        result = np.exp(-(np.linalg.norm(X-Y)**2)/(2*sigma**2))
        return result

    def plot_predict(self, X, alphas, bias, kernel, kernel_parameter):
        '''
        Implements the predict function required for plotting
        predicts the label of the datapoints using alphas
        and bias terms
        '''

        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            result = 0
            for a, support_vectors_label, support_vectors in zip(alphas, self.support_vectors_label, self.support_vectors):
                result += a * support_vectors_label * kernel(X[i],
                support_vectors, kernel_parameter)
            y_pred[i] = result

        final_predictions = y_pred + bias

        return final_predictions

    def predict(self, X, alphas, bias, kernel, kernel_parameter):
        '''
        Implements the predict function
        predicts the label of the datapoints using alphas
        and bias terms
        '''
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            result = 0
            for a, support_vectors_label, support_vectors in zip(alphas, self.support_vectors_label, self.support_vectors):
                result += a * support_vectors_label * kernel(X[i],
                support_vectors, kernel_parameter)
            y_pred[i] = result

        final_predictions = np.sign(y_pred + bias)

        return final_predictions

    def dual_fit(self, X, Y, kernel, C, parameter):
        '''
        Implements dual form of SVM,
        the dual form of SVM is solved
        using cvxopt and alphas are calculated,
        using alphas weight and bias terms are calculated

        Parameters
        ----------------------------------------
        X        : Features
        Y        : Labels
        kernel   : type of kernel to be applied
        C        : Regularization Parameter
        kernel_parameter : The paramters required for a particular
                            kernel

        Returns
        ------------------------------------------
        support_vector_alphas    : The alphas of support vectors
        bias      : The bias term
        '''

        length, number_features = X.shape

        # Creating the kernel gram matrix
        gram_matrix = np.zeros((length, length))

        for i in range(length):
            for j in range(length):
                gram_matrix[i,j] = kernel(X[i], X[j], parameter)

        K = np.outer(Y, Y)*gram_matrix

        # Creating q
        q = -np.ones(length)

        # making vector G
        g_neg_temp = -np.eye(length)
        g_temp = np.eye(length)
        G = np.vstack((g_neg_temp, g_temp))

        # Creating h vector for slack variables
        h_temp = np.zeros(length)
        h_slack = np.ones(length) * C
        h = np.hstack((h_temp, h_slack))

        alphas = cvxopt.matrix(Y, (1,length))
        bias = cvxopt.matrix(0.0)

        svm_solution = cvxopt.solvers.qp(cvxopt.matrix(K), cvxopt.matrix(q),
        cvxopt.matrix(G), cvxopt.matrix(h), alphas, bias)

        all_alphas = np.ravel(svm_solution['x'])

        # setting threshold to get the support vectors
        threshold = 1e-7
        self.support_vectors_alphas = all_alphas > threshold
        indexes = np.arange(len(all_alphas))[self.support_vectors_alphas]
        alphas = all_alphas[self.support_vectors_alphas]
        self.support_vectors = X[self.support_vectors_alphas]
        self.support_vectors_label = Y[self.support_vectors_alphas]

        # Bias term
        bias = 0
        for i in range(len(alphas)):
            bias += self.support_vectors_label[i]
            bias -= np.sum(alphas * self.support_vectors_label *
            gram_matrix[indexes[i], self.support_vectors_alphas])
        bias /= len(alphas)

        return alphas, bias


class LogisticRegression:
    '''

    Class which implements all the
    required logistic Regression implementations
    of Question 1

    '''

    def plotDecisionBoundaryLR(self, X, Y, scoreFn, values, title=""):
        '''
        Function to plot the logistic regression
        decision boundary

        '''
        # Plot the decision boundary. For that, we will asign a score to
        # each point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = max((x_max-x_min)/200., (y_max-y_min)/200.)
        xx, yy = meshgrid(arange(x_min, x_max, h),
                          arange(y_min, y_max, h))
        zz = array([scoreFn(x) for x in c_[xx.ravel(), yy.ravel()]])
        zz = zz.reshape(xx.shape)
        pl.figure()
        CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
        pl.clabel(CS, fontsize=9, inline=1)
        # Plot the training points
        pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
        pl.title(title)
        pl.axis('tight')
        pl.show(block=False)
        time.sleep(4)
        pl.close()

    def count_mistakes(self, Y, Y_pred):
        '''
        Function which counts
        the number of mistakes
        in the given data

        '''
        return np.sum(Y_pred != Y)

    def designMatrix(self, X, order):
        '''
        Function which creates
        X design matrix

        '''
        phi = np.hstack((np.ones((len(X), 1)), X))

        return phi

    def lr_error_function(self, phi, X, Y, W, order, lambda_value):
        '''
        Logistic Regression Error Function

        = ln(1 + exp(-yt))

        '''
        J = 0
        for i in range(0, len(Y)):

            # For polynomial order
            if order > 1:
                wx = np.dot(np.hstack((np.ones(1,), X[i][0], X[i][1], X[i][0]**2, X[i][0]*X[i][1], X[i][1]**2 )) , W)
            # For linear order
            else:
                wx = np.dot(np.hstack((np.ones(1,), X[i])), W)

            J = J + (math.log(1 + np.exp(- (Y[i]*wx))) + lambda_value*(np.dot(W.T, W)))

        return J

    def lr_predict(self, X, Y, W, order):
        '''
        Prediction Function
        that predicts all the labels
        of the given dataset

        '''

        phi = self.designMatrix(X, order)

        if order == 1:
            initial_predictions = np.dot(phi, W)
        else:
            initial_predictions = []
            for i in range(len(X)):
                result = np.dot(np.hstack((np.ones(1,), X[i][0], X[i][1], X[i][0]**2, X[i][0]*X[i][1], X[i][1]**2 )) , W)
                initial_predictions.append(result)


        predictions = np.sign(initial_predictions)

        # Predictions
        # Uncomment this to print all prediction
        # print(predictions)

        return predictions

    def plot_predictionLR(self, x, W, order):
        '''
        Predict function for plotting
        '''
        if order == 1:
            result = np.dot(np.hstack((np.ones(1,), x)), W)
        else:
            result = np.dot(np.hstack((np.ones(1,), x[0], x[1],
                            x[0]**2, x[0]*x[1], x[1]**2 )) , W)

        return result

    def lr_fit(self, X, Y, lambda_value, order):
        '''
        Function which implements minimization
        of the logistic regression loss function
        to find the optimal weights
        which will be used for
        prediction

        Parameters
        ---------------------------------------
        X     : Features - numpy array
        Y     : Labels - numpy array
        lambda_value : l2 regularization parameter
        order : Polynomial basis order

        Returns
        -----------------------------------------
        W : optimal weights which are computed by gradient descent

        '''

        # Creating X matrix based on given order
        phi = self.designMatrix(X, order)

        # Initial starting values of weights
        if order == 1:
            W = np.ones((order + 2))
        else:
            W = np.ones((order + 4))


        objective_function = lambda W: self.lr_error_function(phi, X, Y, W,
        order, lambda_value)

        W = minimize(objective_function, W, method='BFGS')

        print(W)

        # Collecting weights
        W = W.x

        return W


if __name__ == '__main__':


    #############################################
    # Q 1.
    #############################################

    # 1.1 Logistic regression Implementation

    # Implementation of Logistic Regression using
    # minimizers and l2 regualrizer but lambda = 0
    # in this question

    # 1.2 and 1.3 also here
    # just change the required parameters and uncomment
    # the required datasets

    # Read data
    rd = ReadData()

    # Reading train data
    #X_train, Y_train = rd.read_data("data/data_ls_train.csv")
    #X_train, Y_train = rd.read_data("data/data_nls_train.csv")
    X_train, Y_train = rd.read_data("data/data_nonlin_train.csv")

    # Logistic Regression
    log_regress = LogisticRegression()

    # I have implemented logistic
    # Regression using scipy BFGS minimizer

    # Regularizer value, will be zero in 1.1
    #lambda_value = 0

    # For 1.2 for different lambdas, please increase
    # the regularizer parameter here
    lambda_value = 1
    #lambda_value = 0.5
    #lambda_value = 1.5

    # Polynomial basis order
    # For 1.1 we are calculating for order 1
    #order = 1
    # Change it to order = 2 here itself for 1.3
    order = 2

    # Calculating weights using BFGS
    W = log_regress.lr_fit(X_train, Y_train, lambda_value, order)

    # Predicting based on the computed weights
    Y_train_predictions = log_regress.lr_predict(X_train, Y_train, W, order)

    # Calculating misclassified points in the ls training dataset
    misclass_count = log_regress.count_mistakes(Y_train, Y_train_predictions)

    print("The number of mistakes in this training data are ", misclass_count)

    # Plotting for ls train data
    # predictLr (lambda function)
    #predictLR = lambda x : np.dot(np.hstack((np.ones(1,), x)) , W_ls)
    predictLR = lambda x: log_regress.plot_predictionLR(x, W, order)

    print("Plotting for train")
    log_regress.plotDecisionBoundaryLR(X_train, Y_train, predictLR, [0.5], title='LR Train')

    ##############################################################################################
    #X_val, Y_val = rd.read_data("data/data_ls_validate.csv")
    #X_val, Y_val = rd.read_data("data/data_nls_validate.csv")
    X_val, Y_val = rd.read_data("data/data_nonlin_validate.csv")


    # Predicting based on the computed weight W_ls
    Y_val_predictions = log_regress.lr_predict(X_val, Y_val, W, order)

    # Calculating misclassified points in the ls training dataset
    misclass_count = log_regress.count_mistakes(Y_val, Y_val_predictions)

    print("The number of mistakes in the given validate data are ", misclass_count)

    # Plotting for validate data
    # predictLr (lambda function)
    #predictLR = lambda x : np.dot(np.hstack((np.ones(1,), x)) , W_ls)
    predictLR = lambda x: log_regress.plot_predictionLR(x, W, order)

    print("Plotting validate")
    log_regress.plotDecisionBoundaryLR(X_val, Y_val, predictLR, [0.5],
    title='LR Validate')


    ################################################################################################################
    ## Q 2 SVM

    ##############################################################################################################

    ################################################################
    # Q2 Implementing the SVM Primal and Dual form
    #    Prediction function on all training and validation datasets
    #    using different C
    ################################################################


    # Calling the SVM Function
    svm = SVM()

    # Read dataset
    rd = ReadData()

    # Please uncomment the dataset you want to train and test
    X, Y = rd.read_data("data/data_ls_train.csv")
    #X, Y = rd.read_data("data/data_nls_train.csv")
    #X, Y = rd.read_data("data/data_nonlin_train.csv")

    # SVM Calling

    # Regularization Parameter
    C = 0.1

    # Calling the SVM Primal function
    W, bias = svm.primal_fit(X, Y, C)

    # Predicting using the computed parameters
    Y_predictions =  svm.non_kernel_predict(X, W, bias)

    # Calculating the missclassified points
    missclassified = svm.count_mistakes(Y, Y_predictions)

    print("The missclassified points for SVM Primal on training data are: ", missclassified)

    predictSVM = lambda x : np.dot(W, x.T) + bias

    svm.plotDecisionBoundarySVM(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Primal Training')

    ##############################################################################
    # Testing on validation dataset

    X, Y = rd.read_data("data/data_ls_validate.csv")
    #X, Y = rd.read_data("data/data_nls_validate.csv")
    #X, Y = rd.read_data("data/data_nonlin_validate.csv")

    Y_predictions =  svm.non_kernel_predict(X, W, bias)

    # Calculating the missclassified points
    missclassified = svm.count_mistakes(Y, Y_predictions)

    print("The missclassified points for SVM Primal on validation data are: ", missclassified)

    predictSVM = lambda x : np.dot(W, x.T) + bias

    svm.plotDecisionBoundarySVM(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Primal Validation')

    ##########################################################################################
    # For Dual

    # Please uncomment the dataset you want to train and test
    X, Y = rd.read_data("data/data_ls_train.csv")
    #X, Y = rd.read_data("data/data_nls_train.csv")
    #X, Y = rd.read_data("data/data_nonlin_train.csv")


    # Regularization Parameter
    #C = 1
    C = 0.1

    # For Q2 we only want it without any kernel, hence using linear kernel in
    # this question
    kernel = svm.linear_kernel

    kernel_parameter = None

    alphas, bias = svm.dual_fit(X, Y, kernel, C, kernel_parameter)

    # Predicting using the computed parameters
    Y_predictions =  svm.predict(X, alphas, bias, kernel, kernel_parameter)

    # Calculating the missclassified points
    missclassified = svm.count_mistakes(Y, Y_predictions)

    print("The missclassified points for SVM Dual on training data are: ", missclassified)

    predictSVM = lambda x: svm.plot_predict(x, alphas, bias, kernel, kernel_parameter)

    svm.plotDecisionBoundarySVM(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Dual Training')



    ##############################################################################
    # Testing on validation dataset

    X, Y = rd.read_data("data/data_ls_validate.csv")
    #X, Y = rd.read_data("data/data_nls_validate.csv")
    #X, Y = rd.read_data("data/data_nonlin_validate.csv")

    # Predicting using the computed parameters
    Y_predictions =  svm.predict(X, alphas, bias, kernel, kernel_parameter)

    # Calculating the missclassified points
    missclassified = svm.count_mistakes(Y, Y_predictions)

    print("The missclassified points for Linear SVM Dual on Validation data are: ", missclassified)

    predictSVM = lambda x: svm.plot_predict(x, alphas, bias, kernel, kernel_parameter)

    svm.plotDecisionBoundarySVM(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Dual Validation')


    ##########################################################################################

    #  Q3 Kernel SVM
    #    Implementation of dual second order polynomial and guassian kernel

    ###########################################################################################

    print("Kernel svm")
    # Please uncomment the dataset you want to train and test
    X, Y = rd.read_data("data/data_ls_train.csv")
    #X, Y = rd.read_data("data/data_nls_train.csv")
    #X, Y = rd.read_data("data/data_nonlin_train.csv")


    # Regularization Parameter
    #C = 0.01
    #C = 5
    C = 1
    #C = 0.1
    #C = 0.5

    # Kernels
    kernel = svm.polynomial_kernel
    #kernel = svm.guassian_kernel
    #kernel = svm.linear_kernel

    # Parameters required for kernel
    # In case of polynomial kernel its dimensions
    # In case of guassian kernel it is sigma
    # In case of linear kernel pass it None
    #kernel_parameter = 1
    kernel_parameter = None
    #kernel_parameter = 2
    #kernel_parameter = 7
    #kernel_parameter = 0.01
    #kernel_parameter = 10

    # Calling the SVM Dual function
    alphas, bias = svm.dual_fit(X, Y, kernel, C, kernel_parameter)

    # Predicting using the computed parameters
    Y_predictions =  svm.predict(X, alphas, bias, kernel, kernel_parameter)

    # Calculating the missclassified points
    missclassified = svm.count_mistakes(Y, Y_predictions)

    print("The missclassified points for SVM Dual on training data are: ", missclassified)

    predictSVM = lambda x: svm.plot_predict(x, alphas, bias, kernel, kernel_parameter)

    svm.plotDecisionBoundarySVM(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Dual Training')


    ##############################################################################
    # Testing on validation dataset

    X, Y = rd.read_data("data/data_ls_validate.csv")
    #X, Y = rd.read_data("data/data_nls_validate.csv")
    #X, Y = rd.read_data("data/data_nonlin_validate.csv")

    # Predicting using the computed parameters
    Y_predictions =  svm.predict(X, alphas, bias, kernel, kernel_parameter)

    # Calculating the missclassified points
    missclassified = svm.count_mistakes(Y, Y_predictions)

    print("The missclassified points for Kernel SVM Dual on Validation data are: ", missclassified)

    predictSVM = lambda x: svm.plot_predict(x, alphas, bias, kernel, kernel_parameter)

    svm.plotDecisionBoundarySVM(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Kernel Dual Validation')
