#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:
            

        """
        #print "   "        
        pass

    def generateCounts(self, labels):
        out = np.unique(labels, return_counts=True)
        return out[1] 
    
    def calculateEntropy(self, D):
        Dprobs = D / (1. * np.sum(D))+1e-16
        out = np.sum(Dprobs * np.log(Dprobs)) * -1
        return out
    
    def calculateSplitEntropy(self, Dy, Dn):
        nexamples = (np.sum(Dy)+np.sum(Dn)) * 1.
        a = (np.sum(Dy)/nexamples)*(self.calculateEntropy(Dy))
        b = (np.sum(Dn)/nexamples)*(self.calculateEntropy(Dn))
        return a+b
    
    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: dataset
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        scores = np.ones((X.shape[1],)) * np.inf
        for fidx in xrange(X.shape[1]):
            evalres = self.evaluate_numerical_attribute(X[:, fidx], Y)
            scores[fidx] = evalres[1]
        
        minres=self.evaluate_numerical_attribute(X[:, scores.argmin()], Y)
        v, score, Xlidx, Xridx = minres
        #---------End of Your Code-------------------------#
        return v, score, Xlidx,Xridx
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        nclasses=classes.shape[0]
        sidx = np.argsort(feat)
        sf = feat[sidx]
        sY = Y[sidx]
        
        midpoints = (sf[:-1]+sf[1:])/2.
        splitscores = np.zeros((midpoints.shape[0],))
        
        for i, splitpoint in enumerate(midpoints):
            DyClassCounts = self.generateCounts(Y[feat <= splitpoint])
            DnClassCounts = self.generateCounts(Y[feat > splitpoint])
            splitscores[i] = self.calculateSplitEntropy(DyClassCounts, DnClassCounts)
            
        split = midpoints[splitscores.argmin()]
        minscore = splitscores.min()
        Xlidx = np.nonzero(feat <= split)
        Xridx = np.nonzero(feat > split)
        #---------End of Your Code-------------------------#
            
        return split,minscore,Xlidx,Xridx

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=np.round(np.sqrt(nfeatures))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#
        return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#
        return splitvalue, minscore, Xlidx, Xridx

# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#

        return minscore, bXl, bXr


    

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#
        
