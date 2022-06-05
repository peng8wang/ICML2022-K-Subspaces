
function e = TIPS(Z, K, tau)

%%% This is the extension of Thresholding Inner-Product (TIP) Subspace Clustering in 
%%% "Theory of Spectral Method for Union of Subspace-Based Random Geometry Graph" (Li & Gu, 2021) %%% 
    
    %% default parameter setting
    N = size(Z,2); %% the number of data points
    
    %% compute affinity matrix B
    A = zeros(N,N);
    Z1 = abs(Z'*Z); Z1 = Z1 - diag(diag(Z1)); 
    A(Z1 >= tau) = 1;
    A = sparse(A);

    %% apply spectra clustering
    [U1,~] = eigs(A, K);
    e = kmeans(U1, K, 'replicates', 40);  
    
end