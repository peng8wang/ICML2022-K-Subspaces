
function est_labels = SSC_OMP(X, N, spar_MP, nCluster, Redu_dim)
    
    X = dimReduction_PCA(X, Redu_dim);
    R = OMP_mat_func(X, spar_MP, 1/10^20); 
    R(1:N+1:end) = 0;
%     R = cnormalize_inplace(R, Inf);
    A = abs(R) + abs(R)';
    est_labels = SpectralClustering_OMP(A, nCluster, 'Eig_Solver', 'eigs');
%      if N <= 1e4
%        est_labels = SpectralClustering(A, nCluster);
%     else
%        est_labels = SpectralClustering_OMP(A, nCluster, 'Eig_Solver', 'eigs');
%     end
%     A = BuildAdjacency(thrC(A,1), nCluster);
%     est_labels = SpectralClustering(A, nCluster);
    
end