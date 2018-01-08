"""
Demo analysis of the clustering package
"""

# Author: Francesco Casola <fr.casola@gmail.com>

# try to cluster the data above using kmean sklearn
from . import clustering as cl

kkm_means = cl.Cluster_class(n_clusters=NC,algorithm="KKM",verbose=1,sigm_gauss=3)
