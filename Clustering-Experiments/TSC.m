
function [e, U] = TSC(Z, d, K, SC_type, q, dist)

%%% Thresholding-Based Subspace Clustering (TSC) in "Robust Subspace Clustering via Thresholding" (Reinhard & Helmut, 2015) %%% 
    fprintf('********* TSC for Subspace Clustering *********\n');
    if nargin < 5
        q = 5;
    end
    if nargin < 6
        dist = 1;
    end
    
    %% default parameter setting
    N = size(Z,2); %% the number of data points
    
    %% compute affinity matrix B
    Z1 = abs(Z'*Z); Z1 = Z1 - diag(diag(Z1)); B = zeros(N,N);    
    for j = 1:N
        a = Z1(j,:); [~,inx] = maxk(a,q); 
        if dist == 1
            B(inx,j) = exp(-2*acos(Z1(j,inx)));
        else
            Z_inx = Z(:,inx); 
            B(inx,j) = abs(inv(Z_inx'*Z_inx)*Z_inx'*Z(:,j));
        end
    end    
    A = B + B'; 

    %% estimate the number of subspaces by the normalized Laplacian matrix
    if nargin < 3
        de = 1./sqrt(sum(A)); D = diag(de); Lap = eye(N) - D*A*D;
        S = eig(Lap); s = sort(S,'descend'); 
        [~, idx] = max(s(1:N-1)-s(2:N)); K = N - idx;
    end

    %% apply spectra clustering
    CKSym = BuildAdjacency(thrC(A,1));
    if SC_type == 0
       e = SpectralClustering(CKSym, K);
    else
       e = SpectralClustering_OMP(CKSym, K, 'Eig_Solver', 'eigs');
    end
    
    %% compute the bases
    for k = 1:K
        Zk = Z(:, e==k); 
        GU = Zk * Zk';    
        [U{k},~] = eigs(GU,d);  
    end

end


