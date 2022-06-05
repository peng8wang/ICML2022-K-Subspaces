NoisySSC and LRSSC demo 

---------------------------------------------------------
Hi,

Thanks for your interests. If you happen to use this package, please cite the two papers (bibtex entry attached at the back).

Best,
Yu-Xiang.

---------------------------------------------------------
Instructions:

1. You may run "sample_run.m" to see a demo.

2. Note that the ADMM algorithm is guaranteed to converge at a rate similar to other first order methods, but there are a couple of numerical parameters to tune. Try to tune them such that the primal and dual residual is more or less balanced.

3. If you choose to use an increasing augmented penalty (\rho > 1), it will converge as long as \mu is upper bounded theoretically, but when \mu gets too large, it will take a long time.

4. From a practical point of view, the intermediate results (before the algorithm converge) are often good enough. So usually you do not need to set a high numerical accuracy.


----------------------------------------------------------
@inproceedings{wang2013noisy,
  title={Noisy Sparse Subspace Clustering},
  author={Wang, Yu-Xiang and Xu, Huan},
  booktitle={Proceedings of The 30th International Conference on Machine Learning},
  pages={89--97},
  year={2013}
}


@inproceedings{wang2013provable,
  title={Provable Subspace Clustering: When LRR meets SSC},
  author={Wang, Yu-Xiang and Xu, Huan and Leng, Chenlei},
  booktitle={Advances in Neural Information Processing Systems},
  pages={64--72},
  year={2013}
}