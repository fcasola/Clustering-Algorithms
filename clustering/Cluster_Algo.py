"""Clustering-Algorithms

We implement methods capable of clustering datasets according to two families of 
algorithms, specifically using a partitional and propagation-separation approach. 

Relevant literature:

"Global Kernel k-Means Algorithm for Clustering in Feature Space":
    
    http://ieeexplore.ieee.org/document/5033312/
    
"Adaptive Nonparametric Clustering":

    https://arxiv.org/pdf/1709.09102.pdf    

"""

# Author: Francesco Casola <fr.casola@gmail.com>

# imports
import warnings
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform


###############################################################################
# Main class grouping all implementations

class Cluster_class():
    """Partitional and propagation-separation clustering

    1st family: Partitional algorithms
    ----------
        
    Grouping points by optimizing a target objective function.
    Implmented algorithms within this family:
        
        - Standard K-Means (KM): 
            Subcase of kKM with a linear kernel.
        
        - Kernel k-Means (KKM): 
            Clusters in feature space.          
        
        - Global Kernel k-Means (GKKM): 
            kKM with a local search to define initial cluster positions.
            
        - Global Kernel k-Means with Convex Mixture Models (GKKM-CMM):
            optimized boosted version of GkKM to reduce complexity.
                        

    General Parameters: common to all algorithms
    ----------
    
    algorithm: "KM", "KKM", "GKKM", default="GKKM-CMM"
        Specifies the algorithm name. Use one  according to the notation above.
        If not specified, GKKM-CMM will run.
    
    verbose: int, default=0
        Verbosity mode.
        
    n_jobs: int
        Making use of parallel computation where possible.
        
        We make use of the "Parallel" method within the Python "joblib" package.
        As such, if -1 all CPUs are used. If 1 is given, no parallel computing 
        code is used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    
    Parameters for the Partitional algorithms
    ----------
    
    kernel: "lin", default="gauss"
        Type of kernel used in feature space. Default is gaussian.
        Algorithm name "KM" forces the kernel to "lin".
        
    n_clusters: int, optional
        Number of clusters that have to be identified in the dataset.
        If not provided explicitly, a routine will automatically infer it from data.
    
    max_iter: int, optional, default=100
        Maximum number of iterations of the fundamental KKM routine.
    
    tol: float, optional, default=1e-6
        Relative tolerance with respect to the initial clustering error after 
        which the KKM routine declares convergence.
        

    Additional Parameters for the Partitional algorithms
    ----------

    n_init: int, optional, default=100
        Number of times the algorithms KM or KKM run with different centroid
        seeds. The KM and KKM algorithms are the only ones not deterministic.
    
    random_seed: int, optional, default=128
        Seed for the random initialization of the KM and KMM routines, when 
        running in a non-deterministic way.
    
    """
    
    def __init__(self,algorithm="GKKM-CMM",verbose=0,n_jobs=1,kernel="gauss",
                 n_clusters=None,max_iter=100,tol=1e-6,n_init=100,random_seed=128):
        
        # dictionary of all implemented algorithms
        impl_algo = dict(partitional=["KM", "KKM", "GKKM", "GKKM-CMM"],
                         prop_sep=["AWC"])
        
        # assignment to the class properties
        self.impl_algo = Impl_algo
        self.algorithm = algorithm
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.kernel = kernel
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_seed = random_seed
        
    
    def _compute_kernel(X):
        """Computes the Kernel for any partitional algorithm"""
        
        if self.kernel is "lin" or self.algorithm is impl_algo['partitional'][0]:
            # linear kernel will be calculated
            K = np.dot(X,X.T)
        elif self.kernel is "gauss":
            # gaussian kernel will be calculated
            pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
            K = np.exp(-pairwise_sq_dists / s**2)        
        else:
             raise ValueError("Invalid Kernel name.")               
        # return the result 
        return K
        
    def fit(self,X,weights=None):
        """Compute partitional or propagation-separation clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        
        weights : array, [1, n_samples], default=numpy.ones((1,n_samples))
        Weights for the individual points of the dataset. Each partitional 
        algorithm is implemented using the weighted case. The non-weighted 
        scenario can be obtained by having an array of ones.
        """        
        
        #we have 3 cases
        if self.algorithm in self.impl_algo['partitional']:            
            #Compute the Kernel based on the dataset
            K = _compute_kernel(X)            
            
        elif self.algorithm in self.impl_algo['prop_sep']:
        
        else:
            # raise error, not a valid algorithm name
            raise ValueError("Invalid algorithm name.")
        
        
        
        
        
        
        
        
