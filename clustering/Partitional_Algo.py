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
import math as mt
import progressbar 
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
            label_samples = Update_labels.copy()
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

        #return
        return (labels_,cluster_distances_,cluster_error_)
    
    
    def run_W_GKKM(self,X,K,weights):
        """Routine running the Weighted Global Kernel k-Mean algorithm
        
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
        #Initialize cluster labels
        temp_labels = np.zeros((n_samples,1))            
        temp_labels_i = np.zeros((n_samples,1))            
        #Initialize quadratic distances of each sample to its cluster center,
        # in feature space
        temp_dist = np.zeros((n_samples,1))                        
        temp_dist_i = np.zeros((n_samples,1))                        
        
        if self.verbose>=0:
            print(10*'-')
            print("Looking for a solution with %d clusters."%self.n_clusters[-1])
            print(10*'-')
        
        for n_cl in self.n_clusters:
            #output status
            if self.verbose>=0:
                print("Finding optimal solution with %d clusters."%n_cl)

            #if it is the 1-cluster iteration, evaluate the error
            if n_cl==1:
                #assign one point to the cluster 1
                temp_labels[0,0] = 1
                #run the W_KKM algorithm
                temp_labels[:,0],temp_clust_err,temp_iter,temp_dist[:,0] = \
                self._run_W_KKM(temp_labels[:,0],K,weights,n_cl)
            else:
                #Progressbar to have an idea of advancement
                progress = progressbar.ProgressBar()
                # take a for loop over the points. For each point remove it 
                # from the set, run KKM, check whether the error is smaller and 
                # if so take this one as the new solution                
                Old_temp_clust_err = temp_clust_err
                #start progress bar
                iterloop = range(n_samples)
                for i in progress(iterloop):                    
                    temp_labels_i[:,0] = labels_[n_cl-1].copy()
                    temp_labels_i[i,0] = n_cl
                    if self.verbose>1:
                        print("Trying making a singleton cluster with point %d."%i)
                    #run the W_KKM algorithm
                    temp_labels_i[:,0],temp_clust_err,temp_iter,temp_dist_i[:,0] = \
                    self._run_W_KKM(temp_labels_i[:,0],K,weights,n_cl)
                    if temp_clust_err<Old_temp_clust_err:
                        temp_labels[:,0] = temp_labels_i[:,0].copy()
                        temp_dist[:,0] = temp_dist_i[:,0].copy()
                        Old_temp_clust_err,Old_temp_iter = temp_clust_err,temp_iter
                #end of run over all points
                temp_clust_err,temp_iter = Old_temp_clust_err,Old_temp_iter                           
            
            if self.verbose>0:
                print(10*'-')
                print("Solution for %d clusters found. Clustering error: %f"%(n_cl,temp_clust_err))
                print(10*'-')                
            #assignment of the optimal solution
            labels_[n_cl] = temp_labels[:,0].copy()
            cluster_distances_[n_cl] = temp_dist[:,0].copy()
            cluster_error_[n_cl] = [temp_clust_err,temp_iter]

        #return
        return (labels_,cluster_distances_,cluster_error_)







    