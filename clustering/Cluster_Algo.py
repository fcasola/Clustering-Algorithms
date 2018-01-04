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


class Cluster_class():
    """Partitional and propagation-separation clustering

    1st family: Partitional 
        
    Grouping points by optimizing a target objective function.
    Implmented algorithms within this family:
        
        - Standard K-Means (KM): 
            Subcase of kKM with a linear kernel
        
        - Kernel k-Means (kKM): 
            Clusters in feature space            
        
        - Global Kernel k-Means (GkKM): 
            kKM with a local search to define initial cluster positions
            
        - Global Kernel k-Means with Convex Mixture Models (GkKM-CMM):
            optimized boosted version of GkKM to reduce complexity

    General Parameters: common to all algorithms
    ----------
    
    algorithm: Specifies the algorithm name. Use one of the following:
        "KM", "kKM", "GkKM", "GkKM-CMM", according to the notation above.
        If not specified, GkKM-CMM will run.
    
    verbose: int, default 0
        Verbosity mode
        
    n_jobs: int
        Making use of parallel computation where possible.
        
        We make use of the "Parallel" method within the Python "joblib" package.
        As such, if -1 all CPUs are used. If 1 is given, no parallel computing 
        code is used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    
    Parameters for the Partitional algorithms
    ----------
    
    
    
    """
