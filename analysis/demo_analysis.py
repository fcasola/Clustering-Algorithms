"""
Demo analysis of the clustering package.
Within the package, the following algorithms have been implemented:

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
            
The present demo evaluates the algorithms performance on the
Shape sets and the UCI datasets lists on the Clustering benchmark datasets page:
    
        - https://cs.joensuu.fi/sipu/datasets/

"""

# Author: Francesco Casola <fr.casola@gmail.com>

# importing the clustering module (personal module) 
from . import clustering as cl
# importing pandas and other modules to handle datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
# os functionalities
import os
#evaluate NMI using the definition in the Efimov paper
from sklearn.metrics import normalized_mutual_info_score as NMI


"""
PART I: The Shape Set

n_samples: from 240 to 3100
n_features: 2
n_clusters: from 2 to 31

"""

#1 - Loading data
path = r"../data/Shape_Set"
#dictionary of a set of dataframes
set_dataframes = {}
#gathering and loading fles
for fname in os.listdir(path):
    if fname.endswith('.txt'):
        local_fname = os.path.join(path,fname).replace("\\","/")
        set_dataframes[fname.strip('.txt')] = pd.read_csv(local_fname,delimiter='\t', lineterminator='\n', \
                                           names = ["X", "Y", "Labels"])

#2 - plotting dataset
#dataset names
dts_names = list(set_dataframes.keys())
#dataset size
dt_size = len(dts_names)
#approximating to nearest bigger square
perf_sq = np.arange(1,100)
id_plots = np.where(perf_sq**2>=dt_size)[0]
n_plots = perf_sq[id_plots[0]]

#making the plot
plt.figure(num=1,figsize=(10,8))
for i in range(1, n_plots**2):
    plt.subplot(n_plots, n_plots, i) 
    ax = plt.scatter(set_dataframes[dts_names[i-1]]["X"], \
                 set_dataframes[dts_names[i-1]]["Y"], \
                 c=set_dataframes[dts_names[i-1]]["Labels"])
    loc_ax = np.array(plt.axis())
    plt.text(np.mean(loc_ax[:2]),loc_ax[2], dts_names[i-1],
             fontsize=18, ha='center')    
    plt.xlabel("X")
    plt.ylabel("Y")

#3- running partitional algorithms and using evaluating metrics
# Elbow plots will be generated for datasets and without data normalization

#run KKM with and without data normalization
#initializing lists of fitted datasets
KKM_no_norm_labels = {}
KKM_norm_labels = {}
sigma_val_no_norm = [1,3,5,7]
sigma_val_norm = [0.05,0.2,0.3,0.5]
#let's select a subset for this first plot
dts_names_selected = ['Aggregation','Compound','pathbased']

print("KKM algorithm.")
for i in dts_names_selected:
    #starting the loop
    print("Considering the %s dataset"%i)
    #initializing datasets
    KKM_no_norm_labels[i] = []
    KKM_norm_labels[i] = []
    #getting the dataset: NO normalization
    XX = set_dataframes[i].as_matrix(columns=["X","Y"])
    #getting the dataset: normalized
    XX_norm = (XX - XX.mean(axis=0))/(XX.std(axis=0)+sys.float_info.epsilon)    
    
    for sigma in range(len(sigma_val_no_norm)):    
        print("Kernel variance %3.2f (non-norm.), %3.2f (norm.)"%(sigma_val_no_norm[sigma],sigma_val_norm[sigma]))
        #initialize the classes
        KKM_means = cl.Cluster_class(algorithm="KKM",verbose=0, \
                                     sigm_gauss=sigma_val_no_norm[sigma])
        KKM_means_norm = cl.Cluster_class(algorithm="KKM",verbose=0, \
                                          sigm_gauss=sigma_val_norm[sigma])
        #starting the clustering: NO normalization
        print("Non-normalized dataset")
        KKM_means.fit(XX)
        #save
        KKM_no_norm_labels[i].append([KKM_means.labels_,KKM_means.cluster_error_])
        #starting the clustering: normalized
        print("Normalized dataset")
        KKM_means_norm.fit(XX_norm)
        #save    
        KKM_norm_labels[i].append([KKM_means_norm.labels_,KKM_means_norm.cluster_error_])
        

#plotting result for different sigma values, using the normalized and non-normalized dataset

#dataset name
dataset_nm = "Aggregation"
#select a solution for this number of initial clusters
n_clus_sel = 7
#making the plot
plt.figure(num=2,figsize=(10,8))
for i in range(0, 4):
    #getting the dataset: NO normalization
    XX = set_dataframes[dataset_nm].as_matrix(columns=["X","Y"])
    label_loc = KKM_no_norm_labels[dataset_nm][i][0][n_clus_sel]
    #getting the dataset: normalized
    XX_norm = (XX - XX.mean(axis=0))/(XX.std(axis=0)+sys.float_info.epsilon)    
    label_loc_norm = KKM_norm_labels[dataset_nm][i][0][n_clus_sel]
    #plot
    plt.subplot(n_plots, n_plots, 2*i+1) 
    ax = plt.scatter(XX[:,0],XX[:,1],c=label_loc)
    loc_ax = np.array(plt.axis())
    plt.text(np.mean(loc_ax[:2]),loc_ax[2], 'Not. Norm., $\sigma$ = %3.2f'%sigma_val_no_norm[i],
             color='red',fontsize=15, ha='center')    
    plt.xlabel("X")
    plt.ylabel("Y")
    #plot
    plt.subplot(n_plots, n_plots, 2*(i+1)) 
    ax = plt.scatter(XX_norm[:,0],XX_norm[:,1],c=label_loc_norm)
    loc_ax = np.array(plt.axis())
    plt.text(np.mean(loc_ax[:2]),loc_ax[2], 'Norm., $\sigma$ = %3.2f'%sigma_val_norm[i],
             fontsize=15, ha='center')    
    plt.xlabel("X")
    plt.ylabel("Y")
    
    
# making the clustering error plot

# maximum number of clusters used to produce the elbow plot
n_clusters = len(KKM_norm_labels[dataset_nm][0][0].keys())
    
Elbow_plot_no_norm = {}
Elbow_plot_norm = {}

#gathering data
for nm in dts_names_selected[0:2]:
    clutser_err_no_norm = np.zeros((len(sigma_val_no_norm),n_clusters))
    clutser_err_norm = np.zeros((len(sigma_val_norm),n_clusters))
    for sgm in range(len(sigma_val_no_norm)):
        for ncl in range(n_clusters):
            clutser_err_no_norm[sgm,ncl] = KKM_no_norm_labels[nm][sgm][1][ncl+1][0]
            clutser_err_norm[sgm,ncl] = KKM_norm_labels[nm][sgm][1][ncl+1][0]
    Elbow_plot_no_norm[nm] = clutser_err_no_norm
    Elbow_plot_norm[nm] = clutser_err_norm
    
#Producing the elbow plot
dim_subset = len(dts_names_selected)
x_clusters = range(1,1+n_clusters)

plt.figure(num=3,figsize=(10,8))
for i,nm in enumerate(Elbow_plot_no_norm.keys()):
    #get the dataset size
    N_dts = len(set_dataframes[dataset_nm])
    #making the plot
    plt.subplot(2, dim_subset, i+1)
    leg_no_norm = []
    lgd_txt = []
    for sgm in range(len(sigma_val_no_norm)):
        a1, = plt.plot(x_clusters,Elbow_plot_no_norm[nm][sgm,:]/N_dts)
        leg_no_norm.append(a1)
        lgd_txt.append('No Norm., $\sigma$ = %3.2f'%sigma_val_no_norm[sgm])    
    plt.legend(leg_no_norm,lgd_txt,frameon=False)
    plt.xlabel("Number of clusters")
    plt.ylabel("Clust. Err./N")
    plt.title(nm)
    #making the plot
    plt.subplot(2, dim_subset, i+4)
    leg_norm = []
    lgd_txt = []
    for sgm in range(len(sigma_val_norm)):
        a1, = plt.plot(x_clusters,Elbow_plot_norm[nm][sgm,:]/N_dts)
        leg_norm.append(a1)
        lgd_txt.append('Norm., $\sigma$ = %3.2f'%sigma_val_norm[sgm])    
    plt.legend(leg_norm,lgd_txt,frameon=False)
    plt.xlabel("Number of clusters")
    plt.ylabel("Clust. Err./N")
    plt.title(nm)


#looking at sigma = 7 for the pathbased case

nm_dts = "Aggregation"
XX = set_dataframes[nm_dts].as_matrix(columns=["X","Y"])

plt.figure(num=4,figsize=(10,8))
for i in range(1, 7):
    plt.subplot(2, 3, i) 
    label_loc_norm = KKM_norm_labels[nm_dts][-1][0][i]
    ax = plt.scatter(XX[:,0], XX[:,1], c=label_loc_norm)
    loc_ax = np.array(plt.axis())
    plt.text(np.mean(loc_ax[:2]),loc_ax[2], "N cluster: %d"%(i),
             fontsize=18, ha='center')    
    plt.xlabel("X")
    plt.ylabel("Y")


# computing for 
#no normalization
sigma_sel = 3

Clusters_set = dict(Aggregation=7,Compound=6,pathbased=3,spiral=3, \
                  D31=31,R15=15,jain=2,flame=2)

list_algorithms = ["KKM","GKKM-CMM"]
list_algorithms = ["GKKM-CMM"]

Cluster_partitional = {}

for nm_algo in list_algorithms:
    Cluster_partitional[nm_algo] = []
    print(nm_algo)
    for nm_set in Clusters_set.keys():
        print(nm_set)
        XX = set_dataframes[nm_set].as_matrix(columns=["X","Y"])
        #
        n_clust_loc = Clusters_set[nm_set]
        class_no_norm = cl.Cluster_class(algorithm=nm_algo,verbose=0, \
                                          n_clusters=n_clust_loc,sigm_gauss=sigma_sel)
        #starting the clustering: NO normalization
        print("Non-normalized dataset")
        class_no_norm.fit(XX)
        #save results
        Cluster_partitional[nm_algo].append([class_no_norm.labels_,class_no_norm.cluster_error_])
        
        
        
# testing the AWC algorithm


list_lambda = [5,10,5/2,5/2,4,4.5,4,3]

Cluster_prop_sep = {}
for i,nm_set in enumerate(Clusters_set.keys()):
    print(nm_set)
    Cluster_prop_sep[nm_set]=[]
    XX = set_dataframes[nm_set].as_matrix(columns=["X","Y"])
    #
    class_no_norm = cl.Cluster_class(algorithm="AWC",verbose=1,lambda_=list_lambda[i])
    #starting the clustering: NO normalization
    print("Non-normalized dataset")
    class_no_norm.fit(XX)
    #save results
    Cluster_prop_sep[nm_set].append(class_no_norm.labels_)



sel =1

elm = list(Clusters_set.keys())
XX = set_dataframes[elm[sel]].as_matrix(columns=["X","Y"])
N_id_cl = list(Cluster_prop_sep[elm[sel]][0].keys())
lab = Cluster_prop_sep[elm[sel]][0][N_id_cl[0]]

plt.figure(num=5,figsize=(10,8))
ax1 = plt.scatter(XX[:, 0], XX[:, 1],c = lab)
plt.title("Element: %s, N clusters: %d vs %d"%(elm[sel],N_id_cl[0],Clusters_set[elm[sel]]))



elm = list(Clusters_set.keys())
#making the plot
plt.figure(num=5,figsize=(10,8))
for sel in range(0, n_plots**2):
    #loading dataset
    XX = set_dataframes[elm[sel]].as_matrix(columns=["X","Y"])
    #getting predicted number of clusters
    N_id_cl = list(Cluster_prop_sep[elm[sel]][0].keys())
    #getting predicted labels
    lab = Cluster_prop_sep[elm[sel]][0][N_id_cl[0]]
    #getting expected labels
    elab = set_dataframes[elm[sel]].as_matrix(columns=["Labels"])    
    #calculate the NMI
    NMI_loc = NMI(elab.reshape(-1,),lab.reshape(-1,))    
    #plotting
    plt.subplot(n_plots, n_plots, sel+1) 
    ax = plt.scatter(XX[:,0],XX[:,1], c=lab)
    loc_ax = np.array(plt.axis())
    plt.text(np.mean(loc_ax[:2]),loc_ax[2], "NMI %3.2f, Cl %d vs %d"%(NMI_loc,N_id_cl[0],Clusters_set[elm[sel]]),
             fontsize=18, ha='center')    
    plt.xlabel("X")
    plt.ylabel("Y")
plt.suptitle("AWC Algorithm")





list_lambda_part = [0.1,0.5,1,2]

Cluster_prop_sep_sp = {}
XX = set_dataframes["spiral"].as_matrix(columns=["X","Y"])

for lam in list_lambda_part:
    print(lam)
    Cluster_prop_sep_sp[lam]=[]
    #
    class_no_norm = cl.Cluster_class(algorithm="AWC",verbose=1,lambda_=lam)
    #starting the clustering: NO normalization
    class_no_norm.fit(XX)
    #save results
    Cluster_prop_sep_sp[lam].append(class_no_norm.labels_)



elm = list(Clusters_set.keys())



#making the plot
XX = set_dataframes["spiral"].as_matrix(columns=["X","Y"])
plt.figure(num=6,figsize=(10,8))
for sel,lam in enumerate(list_lambda_part):
    #getting predicted number of clusters
    N_id_cl = list(Cluster_prop_sep_sp[lam][0].keys())
    #getting predicted labels
    lab = Cluster_prop_sep_sp[lam][0][N_id_cl[0]]
    #getting expected labels
    elab = set_dataframes["spiral"].as_matrix(columns=["Labels"])    
    #calculate the NMI
    NMI_loc = NMI(elab.reshape(-1,),lab.reshape(-1,))    
    #plotting
    plt.subplot(n_plots, n_plots, sel+1) 
    ax = plt.scatter(XX[:,0],XX[:,1], c=lab)
    loc_ax = np.array(plt.axis())
    plt.text(np.mean(loc_ax[:2]),loc_ax[2], "NMI %3.2f, Cl %d vs %d"%(NMI_loc,N_id_cl[0],Clusters_set["spiral"]),
             fontsize=18, ha='center')    
    plt.xlabel("X")
    plt.ylabel("Y")
plt.suptitle("AWC Algorithm")











    
    
#making the plot
for i in range(1, n_plots**2):
    plt.subplot(n_plots, n_plots, i) 
    ax = plt.scatter(set_dataframes[dts_names[i-1]]["X"], \
                 set_dataframes[dts_names[i-1]]["Y"], \
                 c=set_dataframes[dts_names[i-1]]["Labels"])
    loc_ax = np.array(plt.axis())
    plt.text(np.mean(loc_ax[:2]),loc_ax[2], dts_names[i-1],
             fontsize=18, ha='center')    
    plt.xlabel("X")
    plt.ylabel("Y")    
    


i = "Aggregation"
XX = set_dataframes[i].as_matrix(columns=["X","Y"])
#getting the dataset: normalized
XX_norm = (XX - XX.mean(axis=0))/(XX.std(axis=0)+sys.float_info.epsilon)    

#sigma chosen
sigma_sel = 3
#n of cluster selected
n_clus_sel = 7

lab = KKM_no_norm_labels["Aggregation"][sigma_sel-1][0][n_clus_sel]
plt.figure(num=2,figsize=(10,8))
ax1 = plt.scatter(XX[:, 0], XX[:, 1],c = lab)



fig, plot = plt.subplots(1,2,1)
plot.scatter(XX[:, 0], XX[:, 1])

    
#run GKKM with and without data normalization

#run GKKM-CMM with and without data normalization


#lambda test and NMI for AWC



#KKM_means = cl.Cluster_class(n_clusters=NC,algorithm="KKM",verbose=1,sigm_gauss=3)
kkm_means = cl.Cluster_class(n_clusters=NC,algorithm="AWC",verbose=0,lambda_=5)
kkm_means.fit(XX)
N_cluster_AWC = list(kkm_means.labels_.keys())
print(N_cluster_AWC)

