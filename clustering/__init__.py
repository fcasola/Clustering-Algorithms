"""
Package implementing the following clustering algorithms
        - Standard K-Means (KM): 
            Subcase of KKM with a linear kernel.
        
        - Kernel k-Means (KKM): 
            Clusters in feature space.          
        
        - Global Kernel k-Means (GKKM): 
            kKM with a local search to define initial cluster positions.
            
        - Global Kernel k-Means with Convex Mixture Models (GKKM-CMM):
            optimized boosted version of GkKM to reduce complexity.
	    
        - Adaptive Weights Clustering (AWC): 
            A type of adaptive nonparametric clustering.	    
"""

from .Cluster_Algo import Cluster_class

__all__ = ['Cluster_class']
