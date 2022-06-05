
function e = TIPS(Z, K, tau, SC_type)

%%% This is the extension of Thresholding Inner-Product (TIP) Subspace Clustering in 
%%% "Theory of Spectral Method for Union of Subspace-Based Random Geometry Graph" (Li & Gu, 2021) %%% 
    
    %% default parameter setting
    N = size(Z,2); %% the number of data points
    q = 2;

    %% compute affinity matrix B
    Z1 =  abs(Z'*Z); Z1 = Z1 - diag(diag(Z1));        
    Z2 = Z1; Z2(Z2 < tau) = 0; 
    for j = 1:N
        [a,inx] = maxk(Z1(j,:),q);
        Z2(j,inx) = a; 
    end  
    A = sparse(Z2);

    %% apply spectra clustering
    CKSym = BuildAdjacency(thrC(A,1));
    if SC_type == 0
        e = SpectralClustering(CKSym,K);
    else
        e = SpectralClustering_OMP(CKSym, K, 'Eig_Solver', 'eigs');
    end
end