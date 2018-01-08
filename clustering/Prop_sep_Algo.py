"""Partitional Clustering-Algorithms

Module implementing an algorithm of the propagation-separation approach family.
Specifically:
    - Adaptive Weights Clustering (AWC)

Relevant literature:

"Adaptive Nonparametric Clustering":

    https://arxiv.org/pdf/1709.09102.pdf 
    

"""

# Author: Francesco Casola <fr.casola@gmail.com>

# imports
import warnings
import numpy as np
from joblib import Parallel, delayed
from scipy.special import betainc
import progressbar 

np.warnings.filterwarnings('ignore')
###############################################################################
# Main class defining Propagation-separation implementations

class Prop_sep_class():
    """Propagation-separation approach clustering.
    
    We implement here the following algorithm:
        
        - Adaptive Weights Clustering (AWC): 
            A type of adaptive nonparametric clustering.
            
    This class takes its attributes from the class Cluster_class() within the
    CLuster_Algo module. The user should refer to that class for usage.
    """    

    def __init__(self,Init_pars):
        # assignment to the class properties
        #general
        self.impl_algo = Init_pars.impl_algo
        self.algorithm = Init_pars.algorithm
        self.verbose = Init_pars.verbose
        self.n_jobs = Init_pars.n_jobs
        #propagation-separation
        self.lambda_ = Init_pars.lambda_
        self.a_np = Init_pars.a_np
        self.b_hk = Init_pars.b_hk
        
        
    def _initialize_weights(self,D,n_features):
        """Routine for connectivity matrix initialization in the AWC algorithm.
        Here, the procedure discussed in page 9 of arXiv:1709.09102v1 is used.

        Parameters
        ----------
        
        D: matrix, shape=(n_samples, n_samples)
        Matrix of the pairwise distances for the dataset   

        n_features: int
        Number of features (i.e. dimensionality) of the dataset   
        
        Returns
        ----------
        
        w0ij: array, shape==(n_samples, n_samples)
        Represents the initialization of the connectivity matrix      

        h0: float
        The smallest radius among all h0(xi). 
        See appendix B of arXiv:1709.09102v1

        """        
        
        #minimum number of points within the h0-big sphere
        n0 = 2*n_features+2
        
        #number of samples
        n_samples = D.shape[1]
        
        #check that AWC application is even possible
        if n0 >= n_samples:
            #raise error message
            #the number of samples is too small given the dimensionality of the
            #dataset
            raise ValueError("Insufficient number of samples for AWC.\n \
                  Number of samples is too small given the dimensionality of the dataset")
        else:
        
            #finding the h0(xi) value, sample-dependent h0
            #a sample dependent radius containing at least n0 points 
            #within each xi, besides xi itself
            h0xi = np.sort(D,axis=1)[:,n0-1]
            
            #finding the matrix Mij of the pairwise comparisons max(h0(xi),h0(xj)),
            #used for the initialization of w0ij
            [h0_xi,h0_xj] = np.meshgrid(h0xi,h0xi)
            
            #defining matrix Mij
            Mij = np.maximum(h0_xi,h0_xj)
            
            #initializing the connectivity matrix via an element-wise comparison
            w0ij = (D<=Mij).astype(float)            
            
            #get samllest radius
            h0 = np.min(h0xi)
            
        #returning
        return w0ij,h0,h0xi
        
        
    def _initialize_radii_set(self,D,n_features,h0):
        """Routine determining a list of radii for the AWC algorithm

        Parameters
        ----------
        
        D: matrix, shape=(n_samples, n_samples)
        Matrix of the pairwise distances for the dataset. 

        n_features: int
        Number of features (i.e. dimensionality) of the dataset.  

        h0: float
        The smallest radius among all h0(xi). 
        See appendix B of arXiv:1709.09102v1               
        
        Returns
        ----------
        
        h_list: list of floats
        List containing the sequence of radii to be used in the AWC algorithm.
        
        """
        
        #initializing the final list of radii
        h_list = [h0]

        #verbosity
        if self.verbose>=1:
            print("Initial radius is %f."%np.min(h0))
                
        #maximum possible radius
        h_max = np.max(D)
                
        #while we haven't reached the maximum possible radius
        while np.min(h_list[-1]) < h_max:
        
            #getting the number of points within the sphere of the 
            #previous radius: n(xi,h_km1). A number for each xi
            n_xi_hkm1 = np.sum((D<=h_list[-1]),axis=1)
            
            #estimate the number of points that should be (at most) within
            # the sphere defined by the next radius
            n_xi_hk = np.maximum(np.floor(n_xi_hkm1*self.a_np),n_xi_hkm1+1)
            
            #find the points xi for which the ratio n_xi_hk/n_xi_hkm1 is 
            #maximized. It's the expression B.2 of the paper by Efimov et al.
            xi_max_n = np.argmax(n_xi_hk/n_xi_hkm1)
            
            #list of all possible pairwise distances from xi and relative
            #cumulated number of points
            D_flat_uniq,Cnts_points = np.unique(D[xi_max_n,:],return_counts=True)
    
            #vector containing the number of points within the distances
            #specified in D_flat_uniq (point xi excluded)
            matrix_pts = Cnts_points.cumsum()-1                   
            
            #find the candidate radius hk, corresponding to the case in which 
            # matrix_pts== n_xi_hk[xi_max_n]. If such hk does not exist (e.g. 
            #2 or more points equidistant from xi), the condition B.1
            #is weakened and we take the hk having at least n_xi_hk[xi_max_n] elements   
            if matrix_pts[-1]>=n_xi_hk[xi_max_n]:
                id_hk_candidate = np.where(matrix_pts >= n_xi_hk[xi_max_n])[0][0]
            else:
                id_hk_candidate = len(matrix_pts)-1    
            
            #defining the new candidate hk. b*h_{k-1} is a hard limit
            previous_hk = np.min(h_list[-1])
            hk_candidate = min(D_flat_uniq[id_hk_candidate],self.b_hk*previous_hk)
        
            #produce a warning
            if hk_candidate<=previous_hk:
                warnings.warn("\nRadii sequence not increasing!")
            
            #verbosity
            if self.verbose>=1:
                print("New radius found is %f. Max is %f."%(hk_candidate,h_max))
                
            #adding the new candidate hk to the list
            h_list.append(hk_candidate)
            
        #save biggest radius in case it's not there
        if np.min(h_list[-1]) < h_max:
            h_list.append(h_max)
            #verbosity
            if self.verbose>=1:
                print("New radius found is %f. Max is %f."%(h_list[-1],h_max))
            
        #return
        return h_list
    
    
    def _estimate_N_i_and_j(self,D,hk_l,wij_km1):
        """Routine to estimate the N_i_intersect_j matrix, 
        numerator of eq. 2.1 of the Efimov paper"""
        
        #put zeros on the diagonal to avoid the l=i,j terms
        np.fill_diagonal(wij_km1,0)
        
        #must be within the sphere too
        #(within the sphere AND in the same cluster)
        wij_km1 = wij_km1*(D<=hk_l)
        
        #take all pairwise scalar products, as in section 2.1
        N_i_intersect_j = np.dot(wij_km1,wij_km1.T)
        
        N_i_intersect_j2 = np.dot(wij_km1,(D<=hk_l).T)

        #return
        return N_i_intersect_j,N_i_intersect_j2
    
    def _estimate_N_i_comp_j(self,D,hk_l,wij_km1):
        """Routine to estimate the N_i_complem_j matrix, 
        denominator of eq. 2.1 of the Efimov paper."""
        
        #matrix Glj, with 1s where xl does not belong to a sphere as big as hk_l
        #from xj
        Glj = (D>hk_l).astype(float)
        
        #fill the Glj diagonal with zeros to avoid l=i,j counting
        np.fill_diagonal(Glj,0)
        
        #Defining now the first component of the matrix N_i_complem_j.
        # It's the first term in the defining sum. We call it Pij
        Pij = np.dot(wij_km1,Glj)
        
        #Defining the mass of the complement matrix
        N_i_complem_j = Pij + Pij.T
        
        #return
        return N_i_complem_j    
            

    def _compute_qij(self,D,hk_l,n_features):   
        """Routine to compute the q_ij values.
        Eq. 2.5 of the Efimov paper.
                
        Warning: definition in eq. 2.5 of the paper is incorrect!
        One is supposed to use the regularized incomplete beta-function and 
        not the simple incomplete beta. The regularized incomplete is equivalent
        to the incomplete times the beta function. The expression for q(t) has
        been changed accordingly.
        """

        #defining the t_ij terms
        t_ij = np.divide(D,hk_l)
        
        #the incomplete beta-function
        Beta_inc_eval =  betainc((n_features+1)/2,0.5,1 - np.power(t_ij,2)/4)
    
        #evaluate q_ij (different from the paper due to the regularized express.)
        q_ij = 2*np.divide(1,Beta_inc_eval) - 1
        
        #take the inverse
        q_ij = 1/q_ij
        
        #replace nans
        q_ij[np.isnan(q_ij)] = 0
        
        #return
        return q_ij
    
    
    def _evaluate_KL(self,theta_ij,q_ij):
        """Function evaluating the Kullback-Leibler (KL) divergence
        """
        
        #evaluating the KL divergence (we are using here the symmetrized version)
        KL_ij = (-1)**(theta_ij>q_ij)*(theta_ij-q_ij)*np.log((theta_ij*(1-q_ij))/(q_ij*(1-theta_ij)))        
        
        #replace pathological cases with numbers
        KL_ij = np.nan_to_num(KL_ij)
        
        #return
        return KL_ij
      
         
    def _summarize_results(self,wij_final):
        """Routine that gets the final connectivity matrix and returns the 
        cluster labels.        
        """

        #Dictionary initialization        
        labels_ = {}        
        
        #number of samples
        n_samples = wij_final.shape[0]
        
        #defining final label array
        labels_v = np.zeros((1,n_samples))
        
        #list of indeces
        list_id = np.arange(n_samples)
        
        #initialize a cluster index
        cluster_id = 1
        
        #loop until you looked at all points
        while list_id.shape[0] != 0:
        
            #taking the current point xi
            current_point = list_id[0]
            
            #grouping all wij>0 points in the same group
            xj_same_cluster = np.where(wij_final[current_point,:]==1)[0]
            
            #assignment
            labels_v[0,xj_same_cluster] = cluster_id
            
            #remove such points from the list_id array
            id_remove = np.where(np.in1d(list_id,xj_same_cluster))[0]
            list_id = np.delete(list_id,id_remove)
            
            #advance cluster id
            cluster_id += 1            
            
        #save it into a dictionary
        labels_[cluster_id-1] = labels_v
            
        #return the list of labels. Distances from cluster centers and
        #cluster errors not defined in this algorithm
        return (labels_,None,None)
    
    
    def run_AWC(self,D,n_features):
        """Routine running the Adaptive Weights Clustering (AWC) algorithm
        
        Parameters
        ----------
        
        D: matrix, shape=(n_samples, n_samples)
        Matrix of the pairwise distances for the dataset   

        n_features: int
        Number of features (i.e. dimensionality) of the dataset                 
        
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

        #Initializing the weights matrix wij (page 9 of arXiv:1709.09102v1)
        if self.verbose>=0:
            print("Initializing the connectivity matrix.")
        wij_mat,h0,h0xi = self._initialize_weights(D,n_features)
        
    
        #Initializing the sequence of radii hk used in AWC
        if self.verbose>=0:
            print("Initializing the sequence of radii.")
        #hk_list = self._initialize_radii_set(D,n_features,h0)
        hk_list = self._initialize_radii_set(D,n_features,h0)
        
        #loop over the selected radii
        if self.verbose>=0:
            print("Starting the loop over the sequence of radii.")
            
        #starting a progress bar            
        progress = progressbar.ProgressBar()            
        
        #for id_h,hk_l in enumerate(hk_list[:len(hk_list)-1]):
        for id_h in progress(range(1, len(hk_list))):

            #notify
            if self.verbose>=1:
                print("Loop over the radius %d out of %d."%(id_h,len(hk_list)-1))
            
            #Counting points
            #estimate the N_i_intersect_j matrix, numerator of eq. 2.1.
            #it's a dense intersection (within the sphere AND in the same cluster), and not
            N_i_intersect_j,N_i_intersect_j_loose = self._estimate_N_i_and_j(D,hk_list[id_h-1],wij_mat.copy())
        
            #estimate the N_i_complement_j matrix, denominator of eq. 2.1
            N_i_complem_j = self._estimate_N_i_comp_j(D,hk_list[id_h-1],wij_mat)
        
            #define the mass of the union N_i_uni_j:
            N_i_uni_j = N_i_intersect_j + N_i_complem_j     
            
            #the theta_ij and q_ij terms in the paper. Start with q_ij
            #compute the qij matrix, defined in eq. 2.3
            q_ij = self._compute_qij(D,hk_list[id_h-1],n_features)    
            #removing complicated situations
            q_ij = q_ij.clip(min=0.05, max=0.9)   
            
            #the theta_ij
            theta_ij = np.divide(N_i_intersect_j,N_i_uni_j)
            #handling a bunch of pathological cases
            #handling situations with too many or too few points
            #almost all points at the intersection of two spheres, around xi and xj
            big_theta_ij = theta_ij >= 0.95
            #if not points in the union, theta_ij is zero
            theta_ij[N_i_uni_j == 0] = 0
            #upper and lower clip. Both theta and q must be within [0,1]
            theta_ij = theta_ij.clip(min=0.05, max=0.9)
            q_ij = q_ij.clip(min=0.05, max=0.9)
            #almost no points at the intersection 
            small_theta_ij = theta_ij <= 0.05
            
            #evaluate the KL divergence and compute the test statistics
            KL_ij = self._evaluate_KL(theta_ij,q_ij)
            #the test statistics
            T_ij = N_i_uni_j*KL_ij                
            
            #disconnecting points xi,xj where no points have been found at
            #the intersections of their spheres.
            T_ij[np.logical_or(small_theta_ij, N_i_intersect_j == 0)] = np.inf
            #if theta_ij is very big, then set it to the cutoff lambda
            T_ij[big_theta_ij] = self.lambda_ 
            
            """handling limits"""
            #1 - do not disconnect, in wij_mat, 
            #points that required an h0 bigger than the present radius
            dont_disc = h0xi > hk_list[id_h-1]
            T_ij[dont_disc, :] = np.inf
            T_ij[:, dont_disc] = np.inf
            
            #Matrix N_xixjexcl that shows the number of points around a point xj and within 
            #the same cluster, provided the point is not xi
            #1-Pij: points xj around xi and within the same cluster of xi
            Pij =  (D <= hk_list[id_h-1])*wij_mat
            #2-counting how many and removing xi
            N_ij_xiexcl = (np.sum(Pij, axis = 1) - 1).reshape(1,-1)
            #3-now counting removing xj
            np.fill_diagonal(Pij, False)
            N_xixjexcl = (np.tile(N_ij_xiexcl,(D.shape[0],1)) - Pij).T            
            #don't change if points within intersection are too few
            dont_disc2 = (N_xixjexcl == N_i_intersect_j_loose) * (N_i_intersect_j < 0.5 * N_i_intersect_j_loose)        
            T_ij[np.logical_or(dont_disc2, dont_disc2.T)] = np.nan                        
                
            #applying eq. 2.6 of the paper
            I = (D <= hk_list[id_h]) * (D > 0) * (T_ij != np.inf) * (np.isnan(T_ij) == False) 
            wij_mat[I] = 1 * (T_ij[I] <= self.lambda_)
            
            wij_mat[np.isnan(T_ij)] = 0
            #always put ones on the diagonal (a point is connected to himself)
            np.fill_diagonal(wij_mat, 1)        
        
        #end of the loop
        if self.verbose>=0:
            print("End of the loop. Collecting cluster data.")
        
        #Summary of the results
        labels_,cluster_distances_,cluster_error_ = self._summarize_results(wij_mat)
        
        if self.verbose>=0:
            print("Done.")
        
        #return
        return (labels_,cluster_distances_,cluster_error_)
        