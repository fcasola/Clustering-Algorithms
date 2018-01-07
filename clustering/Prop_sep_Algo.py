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
import math as mt
from scipy.special import betainc, beta
import progressbar 

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
        self.impl_algo = Init_pars.Impl_algo
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
            h0xi = np.sort(D,axis=1)[:,n0]
            
            #finding the matrix Mij of the pairwise comparisons max(h0(xi),h0(xj)),
            #used for the initialization of w0ij
            [h0_xi,h0_xj] = np.meshgrid(h0xi,h0xi)
            
            #defining matrix Mij
            Mij = np.maximum(h0_xi,h0_xj)
            
            #initializing the connectivity matrix via an element-wise comparison
            w0ij = (D<=Mij).astype(float)
            
            #finding the initial radius used in the analysis
            h0 = np.min(h0xi)
            
        #returning
        return w0ij,h0
        
        
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
        
        #number of samples
        n_samples = D.shape[0]
        
        #initializing the final list of radii
        h_list = [h0]

        #verbosity
        if self.verbose>=1:
            print("Initial radius is %f."%h0)

        #list of all possible pairwise distances
        D_flat_uniq = np.unique(D)
        D_dim = D_flat_uniq.shape[0]
        
        #define a matrix telling us, for each xi, how many points can be found
        #within each of the possible distances
        matrix_pts = np.zeros((n_samples,D_dim))
        
        for d in range(D_dim):
            matrix_pts[:,d] = np.sum((D<=D_flat_uniq[d]),axis=1).reshape(-1,)
        
        #maximum possible radius
        h_max = np.max(D_flat_uniq)        
        
        #while we haven't reached the maximum possible radius
        while h_list[-1] < h_max:
        
            #getting the number of points within the sphere of the 
            #previous radius: n(xi,h_km1). A number for each xi
            n_xi_hkm1 = np.sum((D<=h_list[-1]),axis=1)
            
            #estimate the number of points that should be (at most) within
            # the sphere defined by the next radius
            n_xi_hk = np.floor(n_xi_hkm1*self.a_np)
            
            #find the points xi for which the ratio n_xi_hk/n_xi_hkm1 is 
            #maximized. It's the expression B.2 of the paper by Efimov et al.
            xi_max_n = np.argmax(n_xi_hk/n_xi_hkm1)
            
            #find the candidate radius hk, corresponding to the previous increase.
            #if a distance where matrix_pts[xi_max_n,:]== np.max(n_xi_hk) does
            #not exist (e.g. 2 or more points equidistant from xi), the condition B.1
            #is weakened and we take the hk having at least n_xi_hk[xi_max_n] elements            
            id_hk_candidate = np.where(matrix_pts[xi_max_n,:]>=n_xi_hk[xi_max_n])[0][0]
            
            #defining the new candidate hk. b*h_{k-1} is a hard limit
            hk_candidate = np.min(D_flat_uniq[id_hk_candidate],self.b_hk*h_list[-1])
        
            #produce a warning
            if hk_candidate<=h_list[-1]:
                warnings.warn("\nRadii sequence not increasing!")
            
            #verbosity
            if self.verbose>=1:
                print("New radius found is %f. Max is %f."%(hk_candidate,h_max))
                
            #adding the new candidate hk to the list
            h_list.append(hk_candidate)
            
        #return
        return h_list
    
    
    def _estimate_N_i_and_j(self,wij_km1):
        """Routine to estimate the N_i_intersect_j matrix, 
        numerator of eq. 2.1 of the Efimov paper"""
        
        #put zeros on the diagonal to avoid the l=i,j terms
        np.full_diagonal(wij_km1,0)
        
        #take all pairwise scalar products, as in section 2.1
        N_i_intersect_j = np.dot(wij_km1,wij_km1.T)
        
        #return
        return N_i_intersect_j
    
    def _estimate_N_i_comp_j(self,D,hk_l,wij_km1):
        """Routine to estimate the N_i_complem_j matrix, 
        denominator of eq. 2.1 of the Efimov paper."""
        
        #matrix Glj, with 1s where xl does not belong to a sphere as big as hk_l
        #from xj
        Glj = (D>=hk_l).astype(float)
        
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
        
        #defining the argument of the log
        argument_log = np.divide(theta_ij*(1-q_ij),(1-theta_ij)*q_ij)
        
        #evaluating the divergence
        KL_ij = (theta_ij-q_ij)*np.log(argument_log)
        
        #return
        return KL_ij
    
    
    def _heaviside(self,x,val=0.5):
        """Custom definition of the Heaviside function.
        Allows >= vs > and element-wise implementation.
        """
        
        y = np.ones_like(x, dtype=np.float32)
        y[x < 0.0] = 0.0
        y[x == 0.0] = val
        
        #return
        return y    
         
            
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
        wij_mat,h0 = self._initialize_weights(D)
        
        #Initializing the sequence of radii hk used in AWC
        if self.verbose>=0:
            print("Initializing the sequence of radii.")
        hk_list = self._initialize_radii_set(D,n_features,h0)
        
        #loop over the selected radii
        for hk_l in hk_list:
            
            #estimate the N_i_intersect_j matrix, numerator of eq. 2.1
            N_i_intersect_j = self._estimate_N_i_and_j(wij_mat.copy())
        
            #estimate the N_i_complement_j matrix, denominator of eq. 2.1
            N_i_complem_j = self._estimate_N_i_comp_j(D,hk_l,wij_mat)
        
            #define the mass of the union N_i_uni_j:
            N_i_uni_j = N_i_intersect_j + N_i_complem_j
            
            #define the theta_ij matrix in eq. 2.1
            theta_ij = np.divide(N_i_intersect_j,N_i_uni_j)
        
            #compute the qij matrix, defined in eq. 2.3
            q_ij = self._compute_qij(D,hk_l,n_features)    
        
            #evaluating the Kullback-Leibler (KL) divergence
            KL_ij = self._evaluate_KL(theta_ij,q_ij)
        
            #evaluating Heaviside functions
            H_ij = (self._heaviside(q_ij-theta_ij,1) - self._heaviside(theta_ij-q_ij,0))
            
            #evaluating the test-statistics matrix Tij
            T_ij = N_i_uni_j*KL_ij*H_ij

                    
        