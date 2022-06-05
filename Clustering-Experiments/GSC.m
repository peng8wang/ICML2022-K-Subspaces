
function [e, U] = GSC(Z, K, d, SC_type, q)

%%% Nearest Subspace Neighbor + Greedy Subspace Recovery in "Greedy Subspace Clustering" (Parl et al., 2014) %%% 
    fprintf('********* Greedy Subspace Clustering *********\n');
    if nargin < 5
        q = 5;
    end

    %% default parameter setting
    N = size(Z,2); %% the number of data points
    
    %% nearest subspce neighbor (NSN)
    W = zeros(N,N);
    for i = 1:N
       inx_i = i;
       for k = 1:q
           if k <= d
               U0 = orth(full(Z(:,inx_i)));
           end
           if size(U0,2) == 1
               a = abs(U0'*Z);
           else
               UZ = U0'*Z; a = sum(UZ.*UZ,1);
%                a1 = zeros(N,1);
%                inx_j = setdiff(1:N,inx_i);
%                UZ = U0'*Z(:,inx_j); 
%                a1(inx_j) = vecnorm(UZ);
           end
           a(inx_i) = 0; 
           [~,j] = max(a);           
           inx_i = [inx_i j];
       end
       W(i,inx_i) = 1;
    end
    A = W + W';
    
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
        [U{k},~] = eigs(GU, d); 
    end
    
end