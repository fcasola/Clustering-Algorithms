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
from scipy.spatial.distance import pdist, squareform
#personal imports
import Partitional_Algo as pa

###############################################################################
# Main class grouping all implementations

class Cluster_class():
    """Partitional and propagation-separation clustering

    1st family: Partitional algorithms
    ----------
        
    Grouping points by optimizing a target objective function.
    Implmented algorithms within this family:
        
        - Standard K-Means (KM): 
            Subcase of KKM with a linear kernel.
        
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
        Verbosity mode. There are 3 levels of verbosity: 0,1,2.
        
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
        
    sigm_gauss: float, optional, default=1.0
        Sigma of the Gaussian kernel. 
        
    n_clusters: int, optional
        Number of clusters that have to be identified in the dataset.
        If not provided explicitly, a routine will automatically infer it from data.
    
    max_iter: int, optional, default=100
        Maximum number of iterations of the fundamental KKM routine.
    
    tol: float, optional, default=1e-5
        Relative tolerance with respect to the initial clustering error after 
        which the KKM routine declares convergence.
        

    Additional Parameters for the Partitional algorithms
    ----------

    n_init: int, optional, default=100
        Number of times the algorithms KM or KKM run with different centroid
        seeds. The KM and KKM algorithms are the only ones not deterministic.
    
    n_iter_CMM: int, optional, default=500
        Number of iteration during which the location of the exemplars for the
        CMM algorithm does not change.
    
    r_exemplars: int, optional, default=2
        Ratio between the number of exemplars and the total number of clusters.
        Used by the CMM algorithm.
        
    beta_scale: float, optional, default=1
        Constant scaling the parameter beta in the CMM Likelihood
    
    random_seed: int, optional, default=None
        Seed for the random initialization of the KM and KMM routines, when 
        running in a non-deterministic way.
    
    """
    
    def __init__(self,algorithm="GKKM-CMM",verbose=0,n_jobs=1,kernel="gauss",
                 sigm_gauss=1,n_clusters=None,max_iter=100,tol=1e-5,n_init=100,
                 n_iter_CMM=500,r_exemplars=2,beta_scale=1,random_seed=None):
        
        # dictionary of all implemented algorithms
        Impl_algo = dict(partitional=["KM", "KKM", "GKKM", "GKKM-CMM"],
                         prop_sep=["AWC"])
        
        # assignment to the class properties
        self.impl_algo = Impl_algo
        self.algorithm = algorithm
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.kernel = kernel
        self.sigm_gauss = sigm_gauss
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init        
        self.n_iter_CMM = n_iter_CMM
        self.r_exemplars = r_exemplars
        self.beta_scale = beta_scale        
        self.random_seed = random_seed
        
    
    def _compute_kernel(self,X,weights):
        """Computes the Kernel for any partitional algorithm and 
        initializes the weights"""
        
        #kernel
        if self.kernel == "lin" or self.algorithm == self.impl_algo['partitional'][0]:
            # linear kernel will be calculated
            K = np.dot(X,X.T)
        elif self.kernel == "gauss":
            # gaussian kernel will be calculated
            pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
            K = np.exp(-pairwise_sq_dists /(2*self.sigm_gauss**2))        
        else:
             raise ValueError("Invalid Kernel name.")               
        #weights
        if weights is None:
            weights = np.ones((1,X.shape[0]))
        else:
            try:
                if weights.shape[1] != X.shape[0]:
                    raise ValueError    
            except ValueError:    
                print("Invalid array size for the weights.")
                raise             
        # return the result 
        return (K,weights)
    
    def _estimate_clusters(self,X):
        """Checking and/or inferring from data the number of clusters to be
        used in a partitional clustering algorithm"""
        
        if self.n_clusters is None:
            n_cluster_ev = list(map(int,range(1,11)))
            if self.verbose>0:
                print("Number of clusters not specified.\n\
                      The algorithm will run with a 1-10 range, in order to use the Elbow method.")            
        else:
            #check that is a positive integer
            try:
                n_cluster_ev = int(self.n_clusters)
                if n_cluster_ev < 0:
                    raise ValueError  
                #if the algorithm is the GKKM one, then progressively 
                #build the solution up to n_cluster_ev
                if self.algorithm == self.impl_algo['partitional'][2] or \
                self.algorithm == self.impl_algo['partitional'][3]:
                    n_cluster_ev = list(map(int,range(1,n_cluster_ev+1)))
                else:
                    n_cluster_ev = [n_cluster_ev]
            except ValueError:    
                print("Number of clusters must be a positive integer.")
                raise
        #return cluster number
        return n_cluster_ev
            
    def _check_pars(self,Pars_to_check):
        """Checks whether parameters are correctly defined"""
        
        for i in Pars_to_check.keys():            
            try:
                local_pars = Pars_to_check[i][1](Pars_to_check[i][0])
                if local_pars < 0:
                    raise ValueError  
            except ValueError:    
                print("Parameter %s must a positive of %s."%(i,Pars_to_check[i][1]))
                raise        
    
    def _scheduler_partitional(self,K,weights):
        """General scheduler for the partitional algorithms
        Runs the approprate subroutine based on the algorithm name.
        """
        
        #Initialize the object
        partitional = pa.Partitional_class(self)
        
        #Check that parameters have proper values
        if self.random_seed is None:
            check_seed = 0
        else:
            check_seed = self.random_seed
            
        Pars_to_check = dict(verbose = [self.verbose,int], sigm_gauss = [self.sigm_gauss,float],
                     n_init=[self.n_init,int], max_iter=[self.max_iter,int],
                     n_iter_CMM=[self.n_iter_CMM,int],r_exemplars=[self.r_exemplars,int],
                     beta_scale=[self.beta_scale,float],random_seed=[check_seed,int],tol=[self.tol,float])
        self._check_pars(Pars_to_check)
        
        #Running the scheduler
        if self.algorithm == self.impl_algo['partitional'][0]:
            if self.verbose>0:
                print("Running the standard k-Means algorithm.")
            #Run the stand-alone KKM algorithm with a linear kernel
            labels_,cluster_distances_,cluster_error_ = \
            partitional.run_W_KKM_SA(K,weights)
            
        elif self.algorithm == self.impl_algo['partitional'][1]:
            if self.verbose>0:
                print("Running the Kernel k-Means algorithm.")
            #Run the stand-alone KKM algorithm
            labels_,cluster_distances_,cluster_error_ = \
            partitional.run_W_KKM_SA(K,weights)
        
        elif self.algorithm == self.impl_algo['partitional'][2]:
            if self.verbose>0:
                print("Running the Global Kernel k-Means algorithm.")
            #Run the GKKM algorithm
            labels_,cluster_distances_,cluster_error_ = \
            partitional.run_W_GKKM(K,weights)
        
        else:
            if self.verbose>0:
                print("Running the Global Kernel k-Means algorithm with Convex Mixture Models.")
            #Run the GKKM-CMM algorithm
            labels_,cluster_distances_,cluster_error_ = \
            partitional.run_W_GKKM_CMM(K,weights)
            
        return (labels_,cluster_distances_,cluster_error_)
        
    def fit(self,X,weights=None):
        """Compute partitional or propagation-separation clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        
        weights : array, [1, n_samples], default=numpy.ones((1,n_samples))
        Weights for the individual points of the dataset. Each partitional 
        algorithm is implemented using the weighted case. The non-weighted 
        scenario can be obtained by having an array of ones.
        
        Computes the following
        ----------

        self.labels_: dictionary containing sample labels.
        Keys represent the number of clusters for which the algorithm ran or the
        number of clusters found by the algorithm.
        Values are (1,n_samples) arrays containing the labels
            
        self.cluster_distances_:  dictionary containing quadratic distances of each 
        sample to its cluster center, in feature space.
        Keys represent the number of clusters for which the algorithm ran.
        Values are (1,n_samples) arrays containing the centroids        
            
        self.cluster_error_: dictionary containing clustering errors and iterations.
        Keys represent the number of clusters for which the algorithm ran or the
        number of clusters found by the algorithm.
        Values are a tuple containing final clustering error and total number 
        of iterations.
            
        """        
        
        #we have 3 cases
        if self.algorithm in self.impl_algo['partitional']:            
            #Compute the Kernel and initialize weights based on the dataset
            K, weights = self._compute_kernel(X,weights)             
            #Estimate the number of clusters, if not given
            self.n_clusters = self._estimate_clusters(X)
            #Start scheduler for the partitional algorithms
            self.labels_,self.cluster_distances_,self.cluster_error_ = \
                self._scheduler_partitional(K,weights)
            
        elif self.algorithm in self.impl_algo['prop_sep']:
            pass
        else:
            # raise error, not a valid algorithm name
            raise ValueError("Invalid algorithm name.")
        
        return self        
        
        
        
        
        
        
