"""Partitional Clustering-Algorithms

Module implementing several partitional clustering-algorithms.
Specifically:
    - standard k-Means algorithm
    - Kernel k-Means algorithm
    - Global Kernel k-Means algorithm
    - Global Kernel k-Means algorithm with Convex Mixture Models

Relevant literature:

"Global Kernel k-Means Algorithm for Clustering in Feature Space":
    
    http://ieeexplore.ieee.org/document/5033312/
    

"""

# Author: Francesco Casola <fr.casola@gmail.com>

# imports
import warnings
import numpy as np
from joblib import Parallel, delayed
import pdb

###############################################################################
# Main class defining Partitional implementations

class Partitional_class():
    """Partitional clustering
    
    We implement here the following algorithms:
        
        - Standard K-Means (KM): 
            Subcase of KKM with a linear kernel.
        
        - Kernel k-Means (KKM): 
            Clusters in feature space.          
        
        - Global Kernel k-Means (GKKM): 
            kKM with a local search to define initial cluster positions.
            
        - Global Kernel k-Means with Convex Mixture Models (GKKM-CMM):
            optimized boosted version of GkKM to reduce complexity.    
    
    This class takes its attributes from the class Cluster_class() within the
    CLuster_Algo module. The user should refer to that class for usage.
    """    

    def __init__(self,Init_pars):
        # assignment of the class properties from the class Cluster_class()
        self.impl_algo = Init_pars.impl_algo
        self.algorithm = Init_pars.algorithm
        self.verbose = Init_pars.verbose
        self.n_jobs = Init_pars.n_jobs
        self.kernel = Init_pars.kernel
        self.sigm_gauss = Init_pars.sigm_gauss
        self.n_clusters = Init_pars.n_clusters
        self.max_iter = Init_pars.max_iter
        self.tol = Init_pars.tol
        self.n_init = Init_pars.n_init
        self.random_seed = Init_pars.random_seed        


    
    def _run_W_KKM(self,label_samples,Kernel_K,weights_W,n_cl):
        """Routine running the basic Weighted Kernel k-Mean algorithm
        
        Parameters
        ----------
        
        label_samples : array, shape=(1, n_samples), defining initial labels
        for the samples
        
        Kernel_K: matrix, shape=(n_samples, n_samples)
            
        weights_W : array, [1, n_samples], default=numpy.ones((1,n_samples))
        Weights for the individual points of the dataset. Each partitional 
        algorithm is implemented using the weighted case. The non-weighted 
        scenario can be obtained by having an array of ones. 
        
        n_cl: int, number of clusters to be identified
        
        Returns
        ----------
       
        label_samples: (1,n_samples) array containing the final labels.
            
        loc_clust_err: float, final clustering error at this run.
        
        loc_num_iter: int, total number of iterations to meet tolerance.
        
        loc_quadr_dist: (1,n_samples) array containing the squares of the distances
        of each sample to the cluster center, in feature space
                
        """
        
        n_samples = Kernel_K.shape[0]
        #zeroing the clustering error
        loc_clust_err = 0
        #zeroing the local number of iterations
        loc_num_iter = 1
        #saving squared distances of each sample from its cluster center
        #in feature space
        loc_quadr_dist = np.zeros((1,n_samples))
        
        while True:
            
            #keeping error from previous iteration
            loc_clust_err_old = loc_clust_err
            loc_clust_err = 0
            
            #memorize the distance of each sample from each cluster center
            Dist_samples_to_centers = np.zeros((n_samples,n_cl))
            
            for i in range(0,n_cl,1):
                
                #identifying points belonging to cluster i
                id_cluster_i = np.where(label_samples==(i+1))[0]
                
                #find the local weights for each of these points
                cls_i_W = weights_W[0,id_cluster_i]
                
                #computing quadratic distances to each center with the Kernel trick
                #Step1: the kernel diagonal
                Dist_samples_to_centers[:,i] += np.diagonal(Kernel_K)
                #Step2: the cluster centers squared, term independent on the samples location
                [xid,yid] = np.meshgrid(id_cluster_i,id_cluster_i)
                Cls_ctr_sq = np.dot(np.dot(cls_i_W,Kernel_K[yid,xid]),cls_i_W)/(sum(cls_i_W)**2)
                Dist_samples_to_centers[:,i] += Cls_ctr_sq
                #Step3: cross term between each sample and cluster center i
                Dist_samples_to_centers[:,i] -= 2*np.dot(Kernel_K[:,id_cluster_i],cls_i_W)/sum(cls_i_W)
        
                #Saving the quadratic distances 
                loc_quadr_dist[0,id_cluster_i] = Dist_samples_to_centers[id_cluster_i,i] 
                
                #adding to the cluster error only the contribution of the points
                #belonging to cluster i
                loc_clust_err += np.dot(Dist_samples_to_centers[id_cluster_i,i],cls_i_W)
                
            #update cluster elements based on the smallest distance
            Update_labels = np.argmin(Dist_samples_to_centers,axis=1)+1
                
            #we have evaluated the error. let's output it to screen
            if self.verbose>1 and loc_num_iter>1:
                print("At iteration %d the clustering error was %f."%(loc_num_iter,loc_clust_err))
                
            #check for convergence
            if loc_num_iter==1:
                pass
            elif loc_num_iter==2:
                #first iteration, defining relative error and convergence condition
                Delta_error_conv = self.tol*loc_clust_err
            else:
                #check convergence 
                Error_variation = loc_clust_err_old - loc_clust_err
                
                if np.abs(Error_variation) <= Delta_error_conv or loc_num_iter>self.max_iter:
                    break
            
            #update samples label
            label_samples = Update_labels
            #increase iteration counter
            loc_num_iter += 1
            
        #returning
        return (label_samples,loc_clust_err,loc_num_iter,loc_quadr_dist)
        
                

    def run_W_KKM_SA(self,X,K,weights):
        """Routine running the Weighted Kernel k-Mean algorithm
        in a stand-alone fashion
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        
        K: matrix, shape=(n_samples, n_samples)
            
        weights : array, [1, n_samples], default=numpy.ones((1,n_samples))
        Weights for the individual points of the dataset. Each partitional 
        algorithm is implemented using the weighted case. The non-weighted 
        scenario can be obtained by having an array of ones.
        
        Returns
        ----------
       
        labels_: dictionary containing sample labels.
        Keys represent the number of clusters for which the algorithm ran.
        Values are (1,n_samples) arrays containing the labels
            
        cluster_distances_: dictionary containing quadratic distances of each 
        sample to its cluster center, in feature space.
        Keys represent the number of clusters for which the algorithm ran.
        Values are (1,n_samples) arrays containing the centroids
            
        cluster_error_: dictionary containing clustering errors and iterations.
        Keys represent the number of clusters for which the algorithm ran.
        Values are a tuple containing final clustering error and total number 
        of iterations.

        
        """        

        #Dictionary initialization        
        labels_ = {}
        cluster_distances_ = {}
        cluster_error_ = {}
        
        #number of samples
        n_samples = X.shape[0]         
        
        for n_cl in self.n_clusters:
            #output status
            if self.verbose>=0:
                print("Finding optimal solution with %d clusters."%n_cl)
            #Initialize cluster labels
            temp_labels = np.zeros((n_samples,self.n_init))            
            #Initialize quadratic distances of each sample to its cluster center,
            # in feature space
            temp_dist = np.zeros((n_samples,self.n_init))                        
            #Initialize clustering error and total iterations
            temp_clust_err,temp_iter = np.zeros((1,self.n_init)),np.zeros((1,self.n_init))             
            #setting the seed for the random number generator
            np.random.seed(self.random_seed)
            
            #loop over different centroid locations
            for i_ctr in range(self.n_init):
                #selecting random points in the cluster to initialize centroids
                sorted_ind = np.arange(n_samples)
                shuffled_ind = np.random.permutation(sorted_ind)                
                #singleton initialization of the cluster labels
                temp_labels[shuffled_ind[0:n_cl],i_ctr] = sorted_ind[0:n_cl]+1
                #run the W_KKM algorithm
                temp_labels[:,i_ctr],temp_clust_err[0,i_ctr],temp_iter[0,i_ctr],temp_dist[:,i_ctr] = \
                self._run_W_KKM(temp_labels[:,i_ctr],K,weights,n_cl)
                #output error
                if self.verbose>0:
                    print(10*'-')
                    print("At initialization %d the final clustering error was %f."%(i_ctr,temp_clust_err[0,i_ctr]))
                    print(10*'-')
            #selecting optimal solution based on clustering error 
            id_min_err = np.where(temp_clust_err[0,:]==np.min(temp_clust_err[0,:]))[0][0]
            
            if self.verbose>0:
                print(10*'-')
                print("Iteration %d selected. Clustering error: %f"%(id_min_err,temp_clust_err[0,id_min_err]))
                print(10*'-')                
            #assignment of the optimal solution
            labels_[n_cl] = temp_labels[:,id_min_err]
            cluster_distances_[n_cl] = temp_dist[:,id_min_err]
            cluster_error_[n_cl] = [temp_clust_err[0,id_min_err],temp_iter[0,id_min_err]]


        return (labels_,cluster_distances_,cluster_error_)




