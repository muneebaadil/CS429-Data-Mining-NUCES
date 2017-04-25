#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# Main pre-processing class that apply necessary preprocessing
# on the features, such as normlization, etc.

# For modeling non-linear boundaries extra features canbe
# added by appending non-linear terms of the given set of 
# features. For this purpose inherit a class and override
# the process_features function.

import numpy as np
import scipy.stats as stats

class PreProcessing:
    """
        Pre-processing class used to scale and pre-process data 

    """
    def __init__(self, Xtrain):
        """
            records the training statistics and apply on the future test examples
            Input:
                Xtrain: an n x d training matrix from which 
                        the training dataset statisitics are  gathered...


        """
        self.xmin= np.min(Xtrain,axis=0)
        self.xmax=np.max(Xtrain,axis=0)
        self.xmean=np.mean(Xtrain,axis=0)
        self.xstd=np.std(Xtrain,axis=0)

    def process_features(self,X):
        """
            Normalize each feature to lie in the range [0 ,1]

            Input:
            ------

                X= n x d dimensional data matrix

            Returns:
            --------

                normalized X
        """

        #X=(X-self.xmin)/(self.xmax-self.xmin)
        X=(X-self.xmean)/self.xstd

        return X
