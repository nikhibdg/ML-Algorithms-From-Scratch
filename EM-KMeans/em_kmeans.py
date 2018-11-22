import random
import sys
import math
import numpy as np
import scipy
from numpy import *
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from plotGauss2D import *
from random import randrange


#############################
# Mixture Of Gaussians
#############################
# A simple class for a Mixture of Gaussians


class MOG:
    def __init__(self, pi = 0, mu = 0, var = 0):
        self.pi = pi
        self.mu = mu
        self.var = var
    def plot(self, color = 'black'):
        return plotGauss2D(self.mu, self.var, color=color)
    def __str__(self):
        return "[pi=%.2f,mu=%s, var=%s]"%(self.pi, self.mu.tolist(), self.var.tolist())
    __repr__ = __str__

colors = ('blue', 'yellow', 'black', 'red', 'cyan')

def plotMOG(X, param, colors = colors):
    fig = pl.figure()                   # make a new figure/window
    ax = fig.add_subplot(111, aspect='equal')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
    ax.set_ylim(min(x_min, y_min), max(x_max, y_max))
    for (g, c) in zip(param, colors[:len(param)]):
        e = g.plot(color=c)
        ax.add_artist(e)
    plotData(X)
    pl.show()


def plotData(X):

    pl.plot(X[:,0:1].T[0],X[:,1:2].T[0], 'gs')


def varMat(s1, s2, s12 = 0):
    return pl.array([[s1, s12], [s12, s2]])


def randomParams(X, mu, var, pi, m):

    # m is the number of mixtures
    # this function is used to generate random mixture, in your homework you should use EM algorithm to get real mixtures.
    # A random mixture...

    return [MOG(pi[i], mu[i], var[i]) for i in range(m)]


class KMeans:

    def euclid_dist(self, x, y):
        return np.linalg.norm(x - y, axis=1)

    def error_dist(self, x,y):
        return np.linalg.norm(x - y)

    def fit(self, X, k, mu):
        '''
        Function which implements
        K Means

        '''
        n,d = X.shape
        cluster_means_prev = ones((k, d))

        objective_error = self.error_dist(cluster_means_prev, mu)
        cluster_means = mu
        centroids = zeros(n)

        while objective_error != 0:
            for i in range(n):
                centroid_point_dist = self.euclid_dist(X[i], cluster_means)
                cluster_number = argmin(centroid_point_dist)
                centroids[i] = cluster_number

            cluster_means_prev = cluster_means

            cluster_means = ones((k, d))
            cluster_length = []
            for i in range(k):
                cluster_points = []
                for j in range(n):
                    if centroids[j] == i:
                        cluster_points.append(X[j])
                cluster_means[i] = mean(cluster_points, axis=0)
                cluster_length.append(len(cluster_points)/n)
            objective_error = self.error_dist(cluster_means, cluster_means_prev)

        return cluster_means, cluster_length


class CV:

    def k_fold_cv_split(self, k, X):
        split_dataset = np.array_split(X, k)
        return split_dataset

    def avglog(self, X, new_mean, new_cov_matrix,new_prior, m):
        log_likelihood = 0.0
        for i in range(len(X)):
            temp_log_likelihood = 0.0
            for j in range(m):
                temp_log_likelihood += new_prior[j] * multivariate_normal(new_mean[j], new_cov_matrix[j], allow_singular=True).pdf(X[i])

            log_likelihood += np.log(temp_log_likelihood)
        average_logliklihood = log_likelihood / len(X)

        return average_logliklihood

    def copy(self, X):
        Y =[]
        for i in X:
            Y.append(i)
        return Y

    def k_cv(self, k, X, em, m, mu, cov_matrix, prior, covariance_type,
    convergence_parameter, iters, cv_x):
        '''
        Function implements the k fold cross validation
        returns the average of all the k fold average likelihood
        '''

        cv_x_copy = self.k_fold_cv_split(k, X)
        all_likelihoods = []
        for i in range(0, len(cv_x)):

            x = cv_x[i]
            cv_x_copy = self.copy(cv_x)
            cv_x_copy.pop(i)
            X_new = cv_x_copy[0]
            for i in range(1, len(cv_x_copy)):
               X_new  = np.concatenate((X_new, cv_x_copy[i]))
            mu = X_new[random.choice(len(X_new), m, False), :]
            new_prior, new_mean, new_cov_matrix, new_average_logliklihood, new_log_likelihood   = em(X_new, m, mu, cov_matrix, prior,
            covariance_type, convergence_parameter, iters)

            test_one_left_part_likelihood =  self.avglog(x, new_mean,
            new_cov_matrix, new_prior, m)
            all_likelihoods.append(test_one_left_part_likelihood)

        return np.mean(all_likelihoods)


class EM:
    '''
    Class which implements
    Expectation Maximization

    '''

    def fit(self, X, m,  mu, cov_matrix, prior, covariance_type,
            convergence_paramater, iters):
        '''
        Class which optimizes the given
        initial parameters of gaussian mixture
        using expectation and maximization

        Parameters
        ----------------------------------------------------
        X               : Features
        m               : NUmber of clusters
        mu              : Initial mean values
        cov_matrix      : Covariance matrix
        prior           : Prior probabilities
        covariance_type : selects which model to use
                          "full" = using general covariance
                                  matrix
                          "diagonal" = using diagonal matrix
        convergence_parameter : Either log likelihood or
                                number of iterations
        iterations     : Number of iterations to convergence


        Returns
        -----------------------------------------------------
        prior                : maximized prior values
        mu                   : maximized mean mixture model
        cov_matrix           : covariance matrix of mixture model
        average_logliklihood : average likelihood of mixture
                                models.


        '''

        convergence = False
        n, d = X.shape
        old_log_likelihood = 50000
        count = 0

        number_clusters = m

        while convergence is not True:

            # Expectation step
            responsibilities = zeros((n, m))

            for i in range(n):
                for j in range(m):
                    responsibilities[i, j] = prior[j] * multivariate_normal(mu[j], cov_matrix[j],
                                                                            allow_singular=True).pdf(X[i])

            resp = responsibilities

            responsibilities = (resp.T / np.sum(resp, axis=1)).T

            weights = np.sum(responsibilities, axis=0)

            # ----------------------------------------------------
            # Maximization step

            # Prior maximization
            prior = (1. / n) * weights

            # Mean and covariance maximization
            temp_cov_matrix = [eye(d)] * m
            for i in range(number_clusters):
                mu[i] = 1./weights[i] * np.sum(responsibilities[:, i] * X.T, axis = 1).T

                deviation = matrix(X - mu[i])

                temp_cov_matrix[i] = np.array(1 / weights[i] *
                                              np.dot(np.multiply(deviation.T,
                                              responsibilities[:, i]),
                                              deviation))

                # Model selection between full covariance and diagonal matrix
                if covariance_type == "full":
                    cov_matrix[i] = temp_cov_matrix[i]
                else:
                    # for diagonal
                    cov_matrix[i] = eye(d)*diag(temp_cov_matrix[i])


            # calculation of average log Likelihood
            log_likelihood = 0.0
            for i in range(n):
                temp_log_likelihood = 0.0
                for j in range(number_clusters):
                    temp_log_likelihood += prior[j] * multivariate_normal(mu[j], cov_matrix[j], allow_singular=True).pdf(X[i])

                log_likelihood += np.log(temp_log_likelihood)
            average_logliklihood = log_likelihood / n

            # Convergance based on loglikelihood
            if convergence_paramater == "likelihood":
                if abs(log_likelihood - old_log_likelihood) < 1e-7:
                    convergence = True
            elif convergence_paramater == "iterations":
                count += 1
                if count > iters:
                    convergence = True
            else:
                count += 1
                if abs(log_likelihood - old_log_likelihood) < 1e-7:
                    convergence = True
                else:
                    if count > iters:
                        convergence = True

            old_log_likelihood = log_likelihood

        return prior, mu, cov_matrix, average_logliklihood, log_likelihood


if __name__ == '__main__':

    ###############################################

    # Q1
    # EM Implementation on all non-mystery training
    # data

    ################################################

    # Select which dataset to be loaded
    #data = 'data/data_1_small'
    #data = 'data/data_1_large'

    #data = 'data/data_2_small'
    #data = 'data/data_2_large'

    #data = 'data/data_3_small'
    #data = 'data/data_3_large'

    #data = "data/mystery_1"

    data = "data/mystery_2"


    # load data from train files
    X = loadtxt(data+'.txt')

    # Initializing EM Class
    em = EM()


    (n, d) = X.shape

    # Number of gaussian clusters
    m = 2

    # Initial random values of mixture

    # Prior probability
    prior= [1./m] * m

    # Mean
    mu = X[random.choice(n, m, False), :]

    # Covariance matrix
    # Initializing to identity matrix
    # and diagonal with some random
    # number

    cov_matrix = [(eye(d)*4)] * m


    # There is change in two parameters
    # to fit through different models of EM
    # change m for changes in number of clusters
    # change covariance_type = 'full' for covariance matrix
    # for diagonal covariance matrix, send
    # covariance_type = "diagonal"
    covariance_type = "full"
    #covariance_type = "diagonal"


    # Covergence parameter
    # I am using two convergence parameter
    # one using log likelihood, to set that:
    #convergence_paramater = "likelihood"
    #iters = None

    # Using the number of iterations
    #convergence_paramater = "iterations"
    convergence_paramater = "both"
    iters = 100

    # Implementation of EM
    # Call the EM.fit function to
    # obtain optimized parameters
    # of given gaussian mixtures
    pr, meannnn, c, average_logliklihood, log_likelihood = em.fit(X, m, mu, cov_matrix, prior, covariance_type, convergence_paramater, iters)
    print("The log likelihood for {} is: ".format(m), log_likelihood)
    print("The average likelihood for {} is: ".format(m),
    average_logliklihood)
    plotMOG(X, randomParams(X, meannnn, c, pr, m))

    ##########################################################################################################
    # Q 2 Variations
    ##########################################################################################################

    # For question 2.1 to just change the parameter covariance_type =
    # "diagonal"
    # for diagonal matrix implementation. For general covariance matrix
    # change convariance_type = "full".

    ##################################################################################################

    # Q2.2
    # Implementation of K-means
    km = KMeans()
    data = 'data/data_1_small'
    #data = 'data/data_1_large'

    #data = 'data/data_2_small'
    #data = 'data/data_2_large'

    #data = 'data/data_3_small'
    #data = 'data/data_3_large'


    # load data from train files
    X = loadtxt(data+'.txt')

    (n, d) = X.shape
    # Provide the number of gaussian components

    m  = 2
    # Random Mean Initialization
    mu = X[random.choice(n, m, False), :]

    mu, prior = km.fit(X, m, mu)

    # Initialized prior and means using KMeans

    # Initializing of covariance with random values
    cov_matrix = [(eye(d)*4)] * m

    # Implementing EM to get parameters

    # Select which dataset to be loaded

    covariance_type = "full"
    #covariance_type = "diagonal"

    #convergence_paramater = "iterations"
    convergence_paramater = "both"
    iters = 100
    #convergence_paramater = "likelihood"
    #iters = None


    # load data from train files
    X = loadtxt(data+'.txt')

    prior, mu, cov_matrix, average_logliklihood, log_likelihood = em.fit(X, m, mu, cov_matrix, prior, covariance_type, convergence_paramater, iters)


    print("The log likelihood for {} is: ".format(m), log_likelihood)
    plotMOG(X, randomParams(X, mu, cov_matrix, prior, m))


    ############################################################
    ## Q 3 Model Selection

    ############################################################
    ##################################################
    # Q 3.1 For this part of question, the code can be reused
    #########################################################

    #########################################################
    # Q 3.2 Implementation of K fold cross validation
    #########################################################
    # Select which dataset to be loaded
    data = 'data/data_1_small'
    #data = 'data/data_1_large'

    #data = 'data/data_2_small'
    #data = 'data/data_2_large'

    #data = 'data/data_3_small'
    #data = 'data/data_3_large'


    # load data from train files
    X = loadtxt(data+'.txt')

    # Doing cross validation on twice, one when k = 4
    # Another is leave one out cross validation
    # in it k = n - 1
    (n, d) = X.shape
    #k = 4
    k = n - 1

    cv = CV()
    # Function to split dataset in X parts


    for m in [1,2,3,4,5]:
        # Initial random values of mixture

        # Prior probability
        prior= [1./m] * m

        # Mean
        mu = X[random.choice(n, m, False), :]

        # Covariance matrix
        # Initializing to identity matrix
        # and diagonal with some random
        # number

        cov_matrix = [(eye(d)*4)] * m


        # There is change in two parameters
        # to fit through different models of EM
        # change m for changes in number of clusters
        # change covariance_type = 'full' for covariance matrix
        # for diagonal covariance matrix, send
        # covariance_type = "diagonal"
        #covariance_type = "full"
        covariance_type = "diagonal"


        # Covergence parameter
        # I am using two convergence parameter
        # one using log likelihood, to set that:
        #convergence_paramater = "likelihood"
        #iters = None

        # Using the number of iterations
        #convergence_paramater = "iterations"
        convergence_parameter = "both"
        iters = 100

        # Initialize k value

        cv_x = cv.k_fold_cv_split(k, X)
        avergage_kcv_likelihood = cv.k_cv(k, X, em.fit, m, mu, cov_matrix, prior, covariance_type, convergence_parameter, iters, cv_x)

        print("average likelihood for {} is: ".format(m), avergage_kcv_likelihood )
