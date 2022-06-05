# Convergence and Recovery Guarantees of the K-Subspaces Method for Subspace Clustering

This folder contains the MATLAB source codes for the implementation of all the experiments in the paper

"Convergence and Recovery Guarantees of the K-Subspaces Method for Subspace Clustering" (accepted by ICML 2021)
by Peng Wang, Huikang Liu, Anthony Man-Cho So, Laura Balzano.

* Contact: Peng Wang
* If you have any questions, please feel free to contact "peng8wang@gmail.com".

===========================================================================

This package contains 2 experimental tests to output the results in the paper:

* In the folder named Convergence-Experiments, we conduct 3 sets of numerical tests to examine the convergence behavior and recovery performance of the KSS method  in the semi-random UoS model. 

* In the folder named Clustering-Experiments, we conduct experiments to examine the computational efficiency and recovery accuracy of the TIPS initialized KSS method on real datasets. We also compare it with several state-of-the-art methods, which are SSC in Elhamifar & Vidal (2013), SSC solved by OMP in You et al. (2016), TSC in Heckel & Bolcskei (2015), GSC in Park et al. (2014), LRR in Liu et al. (2012), and LRSSC in Wang et al. (2019). In the implementation of SSC, OMP, LRR, and LRSSC, we use the source codes provided by their authors.
