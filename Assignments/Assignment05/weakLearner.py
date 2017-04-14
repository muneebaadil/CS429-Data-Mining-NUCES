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
        self.splitpoint=None
        self.fidx=None

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
        self.splitpoint=v
        self.fidx=scores.argmin()
        #---------End of Your Code-------------------------#
        return v, score, Xlidx,Xridx

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        return True if (X[0, self.fidx] <= self.splitpoint) else False
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
    def __init__(self, nsplits=None, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature
            nrandfeat = number of random features to test for each node
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
        minscoreyet=np.inf 
        minresyet=None
        minfidxyet=-1
        feattest=np.unique(np.random.randint(low=0, high=X.shape[1], size=(self.nrandfeat)))
        
        for fidx in feattest:
            res=self.findBestRandomSplit(X[:, fidx], Y)
            if (res[1]<minscoreyet):
                minscoreyet=res[1]
                minresyet=res
                minfidxyet=fidx

        self.splitpoint, self.fidx=minresyet[0], fidx
        minscore, bXl, bXr=minscoreyet, minresyet[2], minresyet[3]

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

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        splitpoints=np.random.uniform(low=feat.min(), high=feat.max(), size=(self.nsplits))

        splitscores=np.ones_like(splitpoints)*np.inf 
        for i, splitpoint in enumerate(splitpoints):
            DyClassCounts = self.generateCounts(Y[feat <= splitpoint])
            DnClassCounts = self.generateCounts(Y[feat > splitpoint])
            splitscores[i] = self.calculateSplitEntropy(DyClassCounts, DnClassCounts)

        splitvalue, minscore=splitpoints[splitscores.argmin()], splitscores.min()
        Xlidx=np.nonzero(feat<=splitvalue)
        Xridx=np.nonzero(feat>splitvalue)

        #---------End of Your Code-------------------------#
        return splitvalue, minscore, Xlidx, Xridx

    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl)) 
        hr= -np.sum(pr*np.log2(pr)) 

        sentropy = pleft * hl + pright * hr

        return sentropy

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
        randcoeffs=np.random.uniform(low=-3, high=+3, size=(nfeatures+1, self.nsplits))
        mships=(X.dot(randcoeffs[:-1, :])+randcoeffs[-1,:])>=0
        
        splitscores=np.zeros(shape=(self.nsplits,))
        for idx in xrange(self.nsplits):
            splitscores[idx]=self.calculateEntropy(Y, mships[:,idx])
        
        minscore=splitscores.min()
        bXl=np.nonzero(mships[:, splitscores.argmin()])
        bXr=np.nonzero(np.logical_not(mships[:, splitscores.argmin()]))
        
        self.splitpoints=randcoeffs[:, splitscores.argmin()]
        
        return minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        out=X[0,:].dot(self.splitpoints[:-1])+self.splitpoints[-1]
        return True if (out>0) else False
        #return True if (X.dot(self.splitpoint[:-1])>self.splitpoint[-1]) else False
        #---------End of Your Code-------------------------#
        
#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits,nrandfeat=2)
        self.fidx, self.splitpoints=None, None
        pass

    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape
        randfeat=np.random.choice(a=np.arange(0,X.shape[1]), size=(self.nrandfeat,),
            replace=False)
        Xnew=np.zeros((nexamples, 5))

        Xnew[:, 0]=X[:, randfeat[0]]**2
        Xnew[:, 1]=X[:, randfeat[1]]**2
        Xnew[:, 2]=X[:, randfeat[0]]*X[:, randfeat[1]]
        Xnew[:, 3]=X[:, randfeat[0]]
        Xnew[:, 4]=X[:, randfeat[1]]

        randcoeffs=np.random.uniform(low=-3,high=+3, size=(Xnew.shape[1]+1, self.nsplits))

        mships=(Xnew.dot(randcoeffs[:-1, :])+randcoeffs[-1,:])>=0
        splitscores=np.zeros(shape=(self.nsplits,))
        for idx in xrange(self.nsplits):
            splitscores[idx]=self.calculateEntropy(Y, mships[:, idx])

        minscore=splitscores.min()
        bXl=np.nonzero(mships[:, splitscores.argmin()])
        bXr=np.nonzero(np.logical_not(mships[:, splitscores.argmin()]))
        
        self.fidx=randfeat
        self.splitpoints=randcoeffs[:, splitscores.argmin()]

        return minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #print X.shape 
        a,b=self.fidx[0], self.fidx[1]
        out=(((X[0, a]**2)*self.splitpoints[0])+((X[0, b]**2)*self.splitpoints[1])+\
                (X[0, a]*X[0,b]*self.splitpoints[2])+(X[0, a]*self.splitpoints[3])+(X[0, b]*self.splitpoints[4])+\
                (self.splitpoints[5])) >= 0
        return out