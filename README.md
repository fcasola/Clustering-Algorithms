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

Assuming that XX is an [n_samples,n_features] array reprenting, e.g., the *Aggregation Set* in the [Clustering benchmark datasets](https://cs.joensuu.fi/sipu/datasets/), clustering using the AWC algorithm and with the hyperparameter λ =5 can be simply performed via the following code:

```
AWC_clust = cl.Cluster_class(algorithm="AWC",lambda_=5)
AWC_clust.fit(XX)
```
The identified total number of clusters n_clusters and a plot of the clustering result can be created by typing:

```
#getting the total number of clusters identified
n_clusters = next(iter(AWC_clust.labels_))

#plotting the result
fig, plot = plt.subplots()
plot.scatter(XX[:, 0], XX[:, 1], c=AWC_clust.labels_[n_clusters])
plt.title("%d clusters identified by AWC"%n_clusters)
```
The previous code would produce the following output:










