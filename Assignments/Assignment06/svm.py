# ---------------------------------------------#
# -------| Written By: Sibt ul Hussain |-------#
# ---------------------------------------------#

from classifier import *
import pdb


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
        return X.dot(theta)

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
        datacost = 1-(X.dot(theta)*Y)
        datacost[datacost<0] = 0 
        datacost = np.sum(datacost)*(1./(2*X.shape[0]))
        
        regcost = (self.lembda/2.) * (np.sum(np.square(theta)))
        
        cost = datacost + regcost
        return cost

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
        nexamples, nfeats = float(X.shape[0]), X.shape[1]

        # Your Code here
        fullgrads = X*(-1*Y)
        conditionvect = Y*self.hypothesis(X,theta)
        fullgrads[conditionvect[:,0] >= 1, :] = np.zeros((nfeats,))
        datagrad = np.sum(fullgrads, axis=0, keepdims=True) * (1./(2.*nexamples))
        
        reggrad = self.lembda*theta
        grad = datagrad.T+reggrad
        #pdb.set_trace()
        return grad 

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
        self.theta = optimizer.gradient_descent(X, Y, self.cost_function, self.derivative_cost_function)
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
        hypo=self.hypothesis(X, self.theta)[:,0]
        out = np.zeros((X.shape[0],))
        out[hypo>0] = +1
        out[hypo<0] = -1
        return out 