# ---------------------------------------------#
# -------| Written By: Sibt ul Hussain |-------#
# ---------------------------------------------#

from classifier import *


# Note: Here the bias term is considered as the last added feature
class SVM(Classifier):
    ''' Implements the SVM For Classification... '''

    def __init__(self, lembda=0.001):
        """
            lembda= Regularization parameter...
        """
        Classifier.__init__(self, lembda)
        pass

    def hypothesis(self, X, theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''

        # Your Code here
        return       

    def cost_function(self, X, Y, theta):
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
        return

    def derivative_cost_function(self, X, Y, theta):
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
        nexamples = float(X.shape[0])

        # Your Code here
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

        nexamples, nfeatures = X.shape
        
        # Your Code here
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

        # Your Code here
        return