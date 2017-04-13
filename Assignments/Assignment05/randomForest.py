#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

#---------------Instructions------------------#
# Please read the function documentation before
# proceeding with code writing. 

# For randomizing, you will need to use following functions
# please refer to their documentation for further help.
# 1. np.random.randint
# 2. np.random.random
# 3. np.random.shuffle
# 4. np.random.normal 


# Other Helpful functions: np.atleast_2d, np.squeeze()
# scipy.stats.mode, np.newaxis

#-----------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
import tree as tree
import numpy as np
import scipy.stats as stats
from collections import defaultdict

class RandomForest:
    ''' Implements the Random Forest For Classification... '''
    def __init__(self, ntrees=10,treedepth=5,usebagging=False,baggingfraction=0.6,
        weaklearner="Conic",
        nsplits=10,        
        nfeattest=None, posteriorprob=False,scalefeat=True ):        
        """      
            Build a random forest classification forest....

            Input:
            ---------------
                ntrees: number of trees in random forest
                treedepth: depth of each tree 
                usebagging: to use bagging for training multiple trees
                baggingfraction: what fraction of training set to use for building each tree,
                weaklearner: which weaklearner to use at each interal node, e.g. "Conic, Linear, Axis-Aligned, Axis-Aligned-Random",
                nsplits: number of splits to test during each feature selection round for finding best IG,                
                nfeattest: number of features to test for random Axis-Aligned weaklearner
                posteriorprob: return the posteriorprob class prob 
                scalefeat: wheter to scale features or not...
        """

        self.ntrees=ntrees
        self.treedepth=treedepth
        self.usebagging=usebagging
        self.baggingfraction=baggingfraction

        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.nfeattest=nfeattest
        
        self.posteriorprob=posteriorprob
        
        self.scalefeat=scalefeat
        
        pass    
    def findScalingParameters(self,X):
        """
            find the scaling parameters
            input:
            -----------------
                X= m x d training data matrix...
        """
        self.mean=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)
    def applyScaling(self,X):
        """
            Apply the scaling on the given training parameters
            Input:
            -----------------
                X: m x d training data matrix...
            Returns:
            -----------------
                X: scaled version of X
        """
        X= X - self.mean
        X= X /self.std
        return X
    def train(self,X,Y,vX=None,vY=None):
            '''
            Trains a RandomForest using the provided training set..
            
            Input:
            ---------
            X: a m x d matrix of training data...
            Y: labels (m x 1) label matrix

            vX: a n x d matrix of validation data (will be used to stop growing the RF)...
            vY: labels (n x 1) label matrix

            Returns:
            -----------

            '''

            nexamples, nfeatures= X.shape

            self.findScalingParameters(X)
            if self.scalefeat:
                X=self.applyScaling(X)

            self.trees=[]
            
            #-----------------------TODO-----------------------#
            #--------Write Your Code Here ---------------------#
            while len(self.trees)<self.ntrees:
                dtnew=tree.DecisionTree(weaklearner=self.weaklearner, nsplits=self.nsplits, nfeattest=self.nfeattest)
                dtnew.train(X, Y)
                self.trees.append(dtnew)
            #---------End of Your Code-------------------------#
        
    def predict(self, X):
        
        """
        Test the trained RF on the given set of examples X
        
                   
            Input:
            ------
                X: [m x d] a d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z=[]
        
        if self.scalefeat:
            X=self.applyScaling(X)

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        out=np.zeros((X.shape[0], len(self.trees)))
        for i, t in enumerate(self.trees):
            out[:, i]=t.test(X)
        pclasses=stats.mstats.mode(out, axis=1)
        return pclasses[0]
        #---------End of Your Code-------------------------#
        
