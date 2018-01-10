# Clustering-Algorithms
In [*unsupervised learning*](http://www.springer.com/it/book/9780387310732) one wants to obtain (or *learn*) a function or an underlying law describing the dataset without having been previously exposed to labelled examples. The idea is the one of being capable of spotting general patterns directly into the raw data. Related techniques make for instance use of concepts like similarity within the data, in which case we talk about *clustering*, or try to discover latent variables within the input set by projecting data from a higher to a lower dimensional space, known as *dimensionality reduction*.

Here we present a Python package called [clustering](clustering/), containing the implementation of two main families of clustering algorithms: *partitional* and *propagation-separation*. An introduction to these two categories and the related algorithms is contained in the [following summary](summary/20180111_FCasola_2pg_report.pdf).

Specifically, the implemented algorithms within the partitional family are:

  * Standard k-Means (KM): considered as a subcase of KKM with a linear kernel.
  * Kernel k-Means (KKM): performing clustering in feature space.
  * Global Kernel k-Means (GKKM): KKM with a local search to define initial cluster positions.
  * Global Kernel k-Means with Convex Mixture Models (GKKM-CMM): optimized boosted version of GKKM to reduce complexity.

Relevant literature for the partitional family is:

  * ["Global Kernel k-Means Algorithm for Clustering in Feature Space"](http://ieeexplore.ieee.org/document/5033312/), <br />
    by G. F. Tzortzis and A. C. Likas, IEEE Trans. on Neur. Net. **20**, 7 (2009).
    
The implemented algorithms within the propagation-separation family are:
   
  * Adaptive Weights Clustering (AWC): a type of adaptive nonparametric clustering.

Relevant literature for the propagation-separation family is:

  * ["Adaptive Nonparametric Clustering"](https://arxiv.org/abs/1709.09102), <br />
    by K. Efimov and L. Adamyan, arXiv:1709.09102v1 (2017).

The clustering algorithms have been tested over two families of datasets contained in the [Clustering benchmark datasets](https://cs.joensuu.fi/sipu/datasets/). The original data is contained in [this folder](data/). The analysis of the package performance has been carried out in the form of interactive [jupyter notebooks](http://jupyter.org/) and is present in the [analysis](analysis/) folder. For instance, here is a link to the extensive analysis of the [*Shape sets*](analysis/Demo_analysis_shape_set.ipynb).

## Package description

### Prerequisites

- Python 3.6
- [SciPy](http://www.scipy.org/install.html)
- [Progressbar](https://pypi.python.org/pypi/progressbar2/3.18.1)

### Install and basic use
First, do:

```
git clone https://github.com/fcasola/Clustering-Algorithms
```

After appending to the path the *Clustering-Algorithms* folder (see also [here](analysis/Demo_analysis_shape_set.ipynb)), the package can be imported by typing, e.g.:

```
import clustering as cl
```

Assuming that XX is an [n_samples,n_features] numpy array representing, e.g., the *Aggregation Set* in the [Clustering benchmark datasets](https://cs.joensuu.fi/sipu/datasets/), clustering using the AWC algorithm and with the hyperparameter λ =5 can be simply performed via the following code:

```
AWC_clust = cl.Cluster_class(algorithm="AWC",lambda_=5)
AWC_clust.fit(XX)
```
The identified total number of clusters n_clusters and a plot of the clustering result can be created by typing:

```
#importing matplotlib
import matplotlib.pyplot as plt

#getting the total number of clusters identified
n_clusters = next(iter(AWC_clust.labels_))

#plotting the result
fig, plot = plt.subplots()
plot.scatter(XX[:, 0], XX[:, 1], c=AWC_clust.labels_[n_clusters])
plt.title("%d clusters identified by AWC"%n_clusters)
```
The previous code would produce the following output:

![Alt text](summary/aggregation_set.png?raw=true "AWC algorithm applied to the Aggregation set.")

As one can see, the code could identify 7 clusters within the set. The different colors represent different classes.

### Parameters and Attributes

Here the class parameters and default values:

```
Cluster_class(algorithm="GKKM-CMM",verbose=0,n_jobs=1,kernel="gauss",
                 sigm_gauss=1,n_clusters=None,max_iter=100,tol=1e-5,n_init=100,
                 n_iter_CMM=500,r_exemplars=2,beta_scale=1,random_seed=None,
                 lambda_=None,a_np=np.sqrt(2),b_hk=1.95)
```
Definitions:

```
    General Parameters: common to all algorithms
    ----------
    
    algorithm: "KM", "KKM", "GKKM", "AWC", default="GKKM-CMM"
        Specifies the algorithm name. Use one  according to the notation above.
        If not specified, GKKM-CMM will run.
    
    verbose: int, default=0
        Verbosity mode. There are 3 levels of verbosity: 0,1,2.
        
    n_jobs: int (NOT IMPLEMENTED YET)
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
        Constant scaling the parameter beta in the CMM Likelihood.
    
    random_seed: int, optional, default=None
        Seed for the random initialization of the KM and KMM routines, when 
        running in a non-deterministic way.
        
        
    2nd family: Propagation-separation algorithms
    ----------
    
    Identifying clustering structure by checking at different points and for
    different scales on departure from local homogeneity.
    Implmented algorithms within this family:
        
        - Adaptive Weights Clustering (AWC): 
            A type of adaptive nonparametric clustering.
    
    
    Parameters for the Propagation-separation algorithms
    ----------
        
    lambda_: float, optional, default=None
        Value for the tuning parameter lambda, defining the "no gap" threshold
        in the AWC algorithm. It is a float. If not specified, automatic 
        calibration will take place.
        
    a_np: float, optional, default=sqrt(2)
        Value "a" in the paper by Efimov et al., defining the geometric series 
        in the number of neighbors for the construction of the set of radii.
    
    b_hk: float, optional, default=1.95
        Value "b" in the paper by Efimov et al., defining the geometric series 
        in the radii growth for the construction of the set of radii.
```
The Cluster_class.fit method contains the following additional parameters:

```
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        
        weights : array, [1, n_samples], optional, default=numpy.ones((1,n_samples))
        Weights for the individual points of the dataset. Each partitional 
        algorithm is implemented using the weighted case. The non-weighted 
        scenario can be obtained by having an array of ones.
```
At the end of the calculation, the Cluster_class contains the following attributes:

```
        labels_: dictionary containing sample labels.
        Keys represent the number of clusters for which the algorithm ran or the
        number of clusters found by the algorithm.
        Values are (1,n_samples) arrays containing the labels
            
        cluster_distances_:  dictionary containing quadratic distances of each 
        sample to its cluster center, in feature space.
        Keys represent the number of clusters for which the algorithm ran.
        Values are (1,n_samples) arrays containing the centroids        
            
        cluster_error_: dictionary containing clustering errors and iterations.
        Keys represent the number of clusters for which the algorithm ran or the
        number of clusters found by the algorithm.
        Values are a tuple containing final clustering error and total number 
        of iterations.
```
