#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# A Logistic Regression algorithm with regularized weights...

from classifier import *
import numpy as np

#Note: Here the bias term is considered as the last added feature 

class LogisticRegression(Classifier):
    ''' Implements the LogisticRegression For Classification... '''
    def __init__(self, lembda=0.001):        
        """
            lembda= Regularization parameter...            
        """
        Classifier.__init__(self,lembda)                
        
        pass
    def sigmoid(self,arg):
        """
            Compute the sigmoid function 
            Input:
                arg can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """

        # Your Code here
        toreturn = -1 * arg.astype(float)
        toreturn = np.exp(toreturn)
        toreturn += 1
        toreturn = 1 / toreturn 
        return toreturn 
    
    
    def hypothesis(self, X,theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        # Your Code here
        return self.sigmoid(np.dot(X, theta))

        
    def cost_function(self, X,Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''
    
    
        # Your Code here
        m, n = X.shape
        
        termA = Y * np.log(self.sigmoid(np.dot(X, theta)))
        termB = (1 - Y) * np.log(1 - self.sigmoid(np.dot(X, theta)))
        cost = (np.sum(termA + termB)) 
        #print cost 
        datacost = -1 * cost / m; 
        regcost = self.lembda * np.sum(np.square(theta))
        cost = datacost + regcost
        return cost

    def derivative_cost_function(self,X,Y,theta):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        
        # Your Code here
        m, n = X.shape
        toalter = np.dot(X.transpose(), (self.sigmoid(np.dot(X, theta))) - (Y))
        datagradient = toalter / m
        reggradient = self.lembda*theta
        gradient = datagradient + reggradient 
        return gradient 
        return

    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''
        
        # Your Code here 
        # Use optimizer here
        self.theta=optimizer.gradient_descent(X, Y, self.cost_function, self.derivative_cost_function)
        return
    
    def predict(self, X):
        
        """
        Test the trained perceptron classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        
        num_test = X.shape[0]
        
        
        # Your Code here
        pclass=self.sigmoid(X.dot(self.theta)) > .5
        return pclass