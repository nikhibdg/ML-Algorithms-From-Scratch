import numpy as np
import pylab as pl
import time
from scipy.optimize import fmin_bfgs


# Question 1. Implement gradient descent
class first_gradient_descent(object):
    '''

    Class which implements all the functions
    of question 1.
    ------------------------------------
    Contains functions which implements basic
    gradient decent using hard coded derivatives,
    central differences and scipy optimizer.

    '''

    def _central_difference(fx):
        '''
        Function which returns the gradient
        of a given objective function using
        central difference method

        Parameter
        ----------------------------------
        fx : Ojective function

        Returns
        -----------------------------------
        Gradient of the objective function

        '''

        def derivative_df(self, x):
            '''
            Function which returns the gradient
            of a given objective function using
            central difference method

            Parameter
            ----------------------------------
            x : Vector argument

            Returns
            -----------------------------------
            Gradient Vector of the objective function

            '''
            h = 1e-7
            gradient = np.zeros(x.shape)

            xn = len(x)
            fxn2 = fx(x)
            for i in range(0, xn):
                temp_x = x[i]
                x[i] = temp_x + h
                fxn1 = fx(x)
                x[i] = temp_x
                gradient[i] = (fxn1 - fxn2) / h

            return gradient

        return derivative_df

    @_central_difference
    def sphere_function(x) :
        '''
        Function which returns the gradient
        of a given objective function using
        central difference method

        Parameter
        ----------------------------------
        x : Vector argument

        Returns
        -----------------------------------
        Gradient Vector of the objective function

        '''
        return sum([i*i for i in x])

    @_central_difference
    def rosenbrock_function(x):

        '''
        Function which returns the gradient
        of a given objective function using
        central difference method

        Parameter
        ----------------------------------
        x : Vector argument

        Returns
        -----------------------------------
        Gradient Vector of the objective function

        '''
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    def rosenbrock(self, x):

        '''
        Function which implements
        rosenbrock function

        Parameters
        ---------------------------------------------------
        X : Vector arguements for rosenbrock
            function

        Returns
        ---------------------------------------------------
        A scalar value obtained after substituting the given
        vector in this function

        '''

        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    def scipy_test(self, x, fx, df):
        '''
        Function which implements the optimization of
        a given objective function using scipy fmin_bfds

        Parameters
        --------------------------------------
        fx : objective function
        x  : Vector arguement
        df : Gradient of objective function

        Returns
        ------------------------------------------
        Minima values

        '''
        xopt = fmin_bfgs(fx, x, fprime=df)
        return xopt

    def sphere_obj_func(self, X) :

        '''
        Function which implements
        sphere function : x^2 + y^2

        Parameters
        --------------------------------------
        X : Vector arguement which contains
            values of x and y

        Returns
        ------------------------------------------
        A scalar value after substituting the given
        vector in this function

        '''

        return sum([i*i for i in X])

    def derive_rosenbrock_function(self, x):
        '''
        Function which returns the gradient of
        rosenbrock function

        Parameters
        ---------------------------------------------
        X : Vector arguements for rosenbrock function

        Returns
        ---------------------------------------------
        Gradient vector of rosenbrock function

        '''
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)

        return der

    def derivative_sphere_function(self, X):
        '''
        Function which returns the gradient of
        sphere function : x^2 + y^2

        Parameters
        --------------------------------------
        X : Vector arguement which contains
            values for x and y

        Returns
        ------------------------------------------
        Gradient vector

        '''
        return np.multiply(X, 2)

    def basic_grad_desc(self, alpha, threshold, x, deriv_x, obj_func):
        '''
        Function which returns gradient descent
        of a given objective function

        Parameters
        --------------------------------------------------------
        alpha            : Learning rate
        threshold        : If the objective function converges below
                           this value in two successive step, then
                           the algorithm stops
        x                : The vector argument
        deriv_obj_func   : derivative of the objective function
        obj_func         : Objective function whose gradient descent
                          we are finding

        Returns
        -------------------------------------------------------
        Gradient descent of this objective function

        '''

        converge_flag = False
        initial_flag = False

        count = 0
        while not converge_flag:
            x = x - alpha*deriv_x(x)

            # Converge criteria, stoping when the objective fnction
            # values below threshold at two successive steps
            if not initial_flag:
                if obj_func(x) <= threshold:
                    initial_flag = True
                else:
                    pass
            else:
                pass

            if initial_flag:
                if obj_func(x) <= threshold:
                    converge_flag = True
                else:
                    pass
            else:
                pass
            count = count + 1

        return x, count

####################################################################
# Q.2 Linear Basis Function Regression

class linear_basis_regression(object):

    def getData(self, name):
        data = pl.loadtxt(name)
        X = data[0:1].T
        Y = data[1:2].T
        return X, Y

    def new_designMatrix(self, X, order):

        nx = []
        phi_array = []
        for i in X:
            nx.append(i[0])

        phi_array.append(np.ones(len(X)))

        if order == 0:
            return np.array(phi_array)
        else:
            for i in range(1, order + 1):
                phi_array.append([k**i for k in nx])

        phi = np.array(phi_array).T
        return phi

    def designMatrix(self, X, order):

        phi_array = []
        phi_array.append(np.ones(len(X)))

        for i in range(1, order + 1):
            phi_array.append([k**i for k in X])

        phi = np.array(phi_array).T

        return phi

    def mle_weight(self, X, Y, M, temp_X, temp_Y):

        '''
        Function which calculates the Maximum Likelihood
        weight vectors

        Parameters
        --------------------------------------------
        X : X vector
        Y : Y vector
        M : order
        temp_X : X vector in original format
        temp_Y : Y vector in original format

        Return
        ----------------------------------------------------
        w : Weight vector

        '''

        pl.plot(temp_X.T.tolist()[0],temp_Y.T.tolist()[0], 'gs')
        phi = self.designMatrix(X, M)

        temp1 = np.linalg.inv(np.dot(phi.T,phi))
        temp2 = np.dot(temp1, phi.T)

        w = np.dot(temp2, Y)

        pts = [[p] for p in pl.linspace(min(X), max(X), 100)]

        new_phi = self.new_designMatrix(pts, M).T
        if M == 0:
            Yp = pl.dot(w, new_phi.T)
        else:
            Yp = pl.dot(w.T, new_phi)
        pl.plot(pts, Yp.tolist())
        pl.show(block=False)
        time.sleep(3)
        pl.close()

        return w

    def sse_and_derivation(self, X, Y, M, W):

        # SSE Function compute
        X_mod = self.designMatrix(X, M)
        wxy = np.dot(X_mod, W) - Y
        J = np.dot(wxy.T, wxy).flatten()[0]

        # Derivative of SSE
        der_w = np.dot(W.T,X_sq) - np.dot(Y.T,X_mod)

    def _central_difference(fx):
        '''
        Function which returns the gradient
        of a given objective function using
        central difference method

        Parameter
        ----------------------------------
        fx : Ojective function

        Returns
        -----------------------------------
        Gradient of the objective function

        '''

        def derivative_df(self, X, Y, M, W):
            '''
            Function which returns the gradient
            of a given objective function using
            central difference method

            Parameter
            ----------------------------------
            x : Vector argument

            Returns
            -----------------------------------
            Gradient Vector of the objective function

            '''
            h = 1e-7
            x = W
            gradient = np.zeros(x.shape)

            xn = len(x)
            fxn2 = fx(X, Y, M, x)
            for i in range(0, xn):
                temp_x = x[i]
                x[i] = temp_x + h
                fxn1 = fx(X, Y, M, x)
                x[i] = temp_x
                gradient[i] = (fxn1 - fxn2) / h

            return gradient

        return derivative_df

    @_central_difference
    def sse_num_derivative(X, Y, M, W):

        phi_array = []
        phi_array.append(np.ones(len(X)))

        for i in range(1, M + 1):
            phi_array.append([k**i for k in X])

        X_mod = np.array(phi_array).T
        wxy = np.dot(X_mod, W) - Y
        J = np.dot(wxy.T, wxy).flatten()[0]

        return J

    def derivative_of_sse(self, W, Y, X, M):

        X_mod = self.designMatrix(X, M)

        X_sq = np.dot(X_mod.T, X_mod)

        der_w = np.dot(W.T,X_sq) - np.dot(Y.T,X_mod)

        return der_w

    def sse_gd(self, X, Y, M, temp_X, temp_Y, alpha, precision):

        nx = []
        count = 0

        converge_flag = False

        W = np.zeros((M + 1))

        X_mod = self.designMatrix(X, M)

        X_sq = np.dot(X_mod.T, X_mod)

        while not converge_flag:
            last_step_x = W
            W = W - alpha*self.derivative_w(W, Y, X_sq, X_mod)

            if np.linalg.norm(W - last_step_x) < precision:
            #if np.linalg.norm(W) == np.linalg.norm(last_step_x):
            #if count > iters:
                converge_flag = True
            count += 1

        pl.plot(temp_X.T.tolist()[0],temp_Y.T.tolist()[0], 'gs')
        pts = [[p] for p in pl.linspace(min(X), max(X), 100)]

        pts_mod = self.new_designMatrix(pts,M).T
        if M == 0:
            Yp = pl.dot(W, pts_mod.T)
        else:
            Yp = pl.dot(W.T, pts_mod)

        pl.plot(pts, Yp.tolist())
        pl.show(block=False)
        time.sleep(2)
        pl.close()

        return count, W


    def sse_function(self, X, M, Y, W):
        X_mod = self.designMatrix(X, M)
        wxy = np.dot(X_mod, W) - Y
        J = np.dot(wxy.T, wxy).flatten()[0]
        return J

    def derivative_w(self, W, Y, X_sq, X_mod):

        der_w = np.dot(W.T,X_sq) - np.dot(Y.T,X_mod)

        return der_w


    def scipy_linear_regression(self, M, X, Y, temp_X, temp_Y):

        X_mod = self.designMatrix(X, M)

        #initial_guess = np.zeros(X.shape)
        initial_guess = np.zeros((M + 1))
        X_sq = np.dot(X_mod.T, X_mod)
        sse = lambda xi: self.sse_function(X, M, Y, xi)
        dsse = lambda xi : self.derivative_w(xi, Y, X_sq, X_mod)

        W = fmin_bfgs(sse, initial_guess, dsse)


        pl.plot(temp_X.T.tolist()[0],temp_Y.T.tolist()[0], 'gs')
        pts = [[p] for p in pl.linspace(min(X), max(X), 100)]

        pts_mod = self.new_designMatrix(pts,M).T
        if M == 0:
            Yp = pl.dot(W, pts_mod.T)
        else:
            Yp = pl.dot(W.T, pts_mod)

        pl.plot(pts, Yp.tolist())
        pl.show(block=False)
        time.sleep(2)
        pl.close()

    ###################################################################
    # Ridge Regression
    ##################################################################

    def ridge_regression_analytical(self, X, Y, M, temp_X, temp_Y, lambda_parameter):

        pl.plot(temp_X.T.tolist()[0],temp_Y.T.tolist()[0], 'gs')
        phi = self.designMatrix(X, M)

        ridge_coeff = lambda_parameter*np.eye(M + 1)
        temp1 = np.linalg.inv(ridge_coeff + np.dot(phi.T,phi))
        t2 = np.dot(temp1, phi.T)

        w = np.dot(t2, Y)

        print("weight", w, w.shape)
        pts = [[p] for p in pl.linspace(min(X), max(X), 100)]

        tem = self.new_designMatrix(pts, M).T
        if M == 0:
            Yp = pl.dot(w, tem.T)
        else:
            Yp = pl.dot(w.T, tem)
        pl.plot(pts, Yp.tolist())
        pl.show(block=False)
        time.sleep(2)
        pl.close()

        return w

    def ridge_derivative_w(self, W, Y, X_sq, X_mod, lambda_parameter):

        der_w = np.dot(W.T,X_sq) - np.dot(Y.T,X_mod) + np.multiply(W, lambda_parameter)

        return der_w


    def ridge_regression_gd(self, X, Y, M, temp_X, temp_Y, alpha, precision,
    lambda_parameter):

        nx = []
        count = 0

        converge_flag = False

        W = np.zeros((M + 1))

        X_mod = self.designMatrix(X, M)

        X_sq = np.dot(X_mod.T, X_mod)

        while not converge_flag:
            last_step_x = W
            W = W - alpha*self.ridge_derivative_w(W, Y, X_sq, X_mod,
            lambda_parameter)

            if np.linalg.norm(W - last_step_x) < precision:
                converge_flag = True
            count += 1


        pl.plot(temp_X.T.tolist()[0],temp_Y.T.tolist()[0], 'gs')
        pts = [[p] for p in pl.linspace(min(X), max(X), 100)]

        pts_mod = self.new_designMatrix(pts,M).T
        if M == 0:
            Yp = pl.dot(W, pts_mod.T)
        else:
            Yp = pl.dot(W.T, pts_mod)

        pl.plot(pts, Yp.tolist())
        pl.show(block=False)
        time.sleep(2)
        pl.close()

        return count, W


    def sse_function(self, X,  M, Y, W):
        X_mod = self.designMatrix(X, M)
        wxy = np.dot(X_mod, W) - Y
        J = np.dot(wxy.T, wxy).flatten()[0]
        return J


if __name__ == '__main__':

    ###############################################################
    # Q 1.
    ###############################################################

    # 1. Basic Gradient descent on scalar function

    #1.1 and 1.2
    first_gd = first_gradient_descent()

    # Vector argument to the function
    x = np.array([0.3, 0.7])
    #x = np.array([0.3, 0.7, 0.2, 0.1, 0.01])

    #alpha
    alpha = 0.01

    threshold = 0

    # Test function f(x) = x^2 + y^2  whose gradient descent we will find

    optimized_fx, count = first_gd.basic_grad_desc(alpha, threshold, x,
    first_gd.derivative_sphere_function, first_gd.sphere_obj_func)

    print("The gradient descent of sphere function is reached in {} iterations on hard coded derivative function is: ".format(count), optimized_fx)
    print("With step size {} and threshold {}".format(alpha, threshold))

    print("The objective value at this global minima is ",
    first_gd.sphere_obj_func(optimized_fx))

    ## Test function for rosenbrock

    x = np.array([0.3, 0.7, 0.2, 0.1, 0.01])

    #alpha
    alpha = 0.001

    threshold = 5.40e-27

    optimized_fx, count = first_gd.basic_grad_desc(alpha, threshold, x,
    first_gd.derive_rosenbrock_function, first_gd.rosenbrock)

    print("\nThe gradient descent of rosenbrock function is reached in {} iterations on hard coded derivative function is: ".format(count), optimized_fx)
    print("With step size {} and threshold {}".format(alpha, threshold))

    print("The objective value at this global minima is ",
    first_gd.rosenbrock(optimized_fx))

    ###############################################################

    # 1.3 Finite Difference


    # Numerical gradient using central difference on sphere function
    x = np.array([0.3, 0.7])
    cent_diff_deriv = first_gd.sphere_function(x)
    #cent_diff_deriv = first_gd.rosenbrock_function(x)

    print("\nNumerical gradient of sphere function using central differece", cent_diff_deriv)

    # Analytical gradient of sphere function
    normal_deriv = first_gd.derivative_sphere_function(x)
    print("Analytical gradient of sphere function", normal_deriv)


    # Numerical gradient using central difference on rosenbrock function
    x = np.array([0.3, 0.7, 0.2, 0.1, 0.01])

    cent_diff_deriv = first_gd.rosenbrock_function(x)

    print("\nNumerical gradient of rosenbrock function using central differece", cent_diff_deriv)

    # Analytical gradient of rosenbrock function
    normal_deriv = first_gd.derive_rosenbrock_function(x)
    print("Analytical gradient of rosenbrock function", normal_deriv)

    #########################################################

    # 1.4 Scipy test

    # For sphere function
    x = np.array([0.3, 0.7])
    xopt = first_gd.scipy_test(x, first_gd.sphere_obj_func,
    first_gd.derivative_sphere_function)

    print("\nscipy test for sphere function", xopt)

    # For rosenbrock function
    x = np.array([0.3, 0.7, 0.2, 0.1, 0.01])
    xopt = first_gd.scipy_test(x, first_gd.rosenbrock,
    first_gd.derive_rosenbrock_function)

    print("\nscipy test for rosenbrock function", xopt)

    #################################################################

    # Q2. Linear Basis Function Regression

    ################################################################

    lbr = linear_basis_regression()

    # Getting data from dataset
    X, Y = lbr.getData("curvefitting.txt")
    temp_X = X
    temp_Y = Y

    # Reshaping the data
    X = X.reshape(10,)
    Y = Y.reshape(10,)


    ####################################################################

    #Q 2.1
    #Calculating and plotting MLE for different
    # values orders
    order = [0, 1, 3, 9]
    for M in order:
        weights = lbr.mle_weight(X, Y, M, temp_X, temp_Y)
        print("MLE weights of order {} is ".format(M), weights)


    #########################################################################

    # Q 2.2
    # SSE Function and its derivative, then comparision
    # with numerical gradient function computed values

    weight_dict = {0:np.array([0.19]), 1: np.array([0.82, -1.27]),
    3:np.array([0.31, 7.99, -25.43, 17.37])}

    for k,v in weight_dict.items():
        d = lbr.sse_num_derivative(X, Y, k, v)
        print("\nNumerical derivative of order {} is ".format(k), d)
        d1 = lbr.derivative_of_sse(v, Y, X, k)
        print("Analytical derivative of order {} is".format(k), d1)

    ############################################################################

    M = [0, 1, 3]

    for i in M:
        alpha = 0.01
        precision = 1e-8
        iters, weight = lbr.sse_gd(X, Y, i, temp_X, temp_Y, alpha, precision)
        scipy_w = lbr.scipy_linear_regression(i, X, Y, temp_X, temp_Y)

        print("For order {} and alpha {} and precision {} converged in {} iterations".format(i, alpha, precision, iters))

        alpha = 0.05
        iters, weight = lbr.sse_gd(X, Y, i, temp_X, temp_Y, alpha, precision)
        scipy_w = lbr.scipy_linear_regression(i, X, Y, temp_X, temp_Y)
        print("For order {} and alpha {} and precision {} converged in {} iterations".format(i, alpha, precision, iters))
        print("Using scipy for order {} the weight vectors are {}".format(i,scipy_w))

    ####################################################################################

    M = [0, 1, 3, 9]
    alpha = 0.05
    precision = 1e-8
    for i in M:
        lambda_parameter = 0.1
        w = lbr.ridge_regression_analytical(X, Y, i, temp_X, temp_Y, lambda_parameter)
        print("For order {} and lambda {}, the gradient descent analytically is {}".format(i, lambda_parameter, w))
        iters, w = lbr.ridge_regression_gd(X, Y, i, temp_X, temp_Y, alpha, precision,  lambda_parameter)

        print("For order {}, alpha {},lambda {} and precision {} converged in {} iterations, the weight vector is {}".format(i, alpha,
        lambda_parameter, precision, iters,
        w))
        lambda_parameter = 0.5
        iters, w = lbr.ridge_regression_gd(X, Y, i, temp_X, temp_Y, alpha, precision,  lambda_parameter)
        print("For order {}, alpha {},lambda {} and precision {} converged in {} iterations, the weight vector is {}".format(i, alpha,
        lambda_parameter, precision, iters, w))

    #######################################################################################
    alpha = 0.05
    precision = 1e-8

    X_trainA, Y_trainA = lbr.getData("regressA_train.txt")

    temp_X_trainA = X_trainA
    temp_Y_trainA = Y_trainA


    # Reshaping the data
    X_trainA = X_trainA.reshape(10,)
    Y_trainA = Y_trainA.reshape(10,)

    iters, weight = lbr.sse_gd(X_trainA, Y_trainA, 1, temp_X_trainA,
    temp_Y_trainA, alpha, precision)

    error = lbr.sse_function(X_trainA, 1, Y_trainA, weight)
    print("error in case of regress A dataset", error)

    print("For order {} and alpha {} and precision {} converged in {} iterations with weights as {}".format(1, alpha, precision, iters, weight))


    X_trainB, Y_trainB = lbr.getData("regressB_train.txt")

    temp_X_trainB = X_trainB
    temp_Y_trainB = Y_trainB


    # Reshaping the data
    X_trainB = X_trainB.reshape(10,)
    Y_trainB = Y_trainB.reshape(10,)

    lambda_parameter = 0.5
    weight = lbr.ridge_regression_analytical(X_trainB, Y_trainB, 6,
    temp_X_trainB, temp_Y_trainB, lambda_parameter)

    error = lbr.sse_function(X_trainB, 6, Y_trainB, weight)
    print("error in case of regress B dataset", error)

    print("For order {} and alpha {} and precision {} converged  with weights as {}".format(6, alpha, precision, weight))


    X_test, Y_test = lbr.getData("regress_validate.txt")
    temp_X_test = X_test
    temp_Y_test = Y_test


    # Reshaping the data
    X_test = X_test.reshape(20,)
    Y_test = Y_test.reshape(20,)

    lambda_parameter = 0.5
    weight = lbr.ridge_regression_analytical(X_test, Y_test, 6,
    temp_X_test, temp_Y_test, lambda_parameter)

    error = lbr.sse_function(X_test, 6, Y_test, weight)
    print("error in case of test dataset", error)

    print("For order {} and alpha {} and precision {} converged  with weights as {}".format(6, alpha, precision, weight))

