

function est_labels = EnKSS(X, K, d, true_labels, B, q, T, heur, sc, base, opts)
    
% The Ensemble K-Subspaces algorithm for subspace clustering
%   [trueErr,l2Err,estLabels] = EKSS(X,K,d,trueLabels,B,q,heur,sc,base)
% Input:
%   X: ambient dimension by number of data points
%   K: number of subspaces
%   d: dimension of subspaces (assumed all equal)
%   trueLabels: vector of true labels
%   B: number of base clusterings
%   q: threshold parameter (default is no thresholding)
%   T: number of KSS iterations (default 30, set to 0 for EKSS-0)
%   heur: set to 1 to add a value besides 1 for co-clustered points (default 0)
%   sc: select whether to run spectral clustering or not (default 1)
%   base: base clustering algorithm, choose from 'kss' and 'copKSS' (default 'kss')
% Output:
%   err: clustering error
%   estLabels: estimated labels
%   A: affinity matrix
%--------------------------------------------------------------------------
% Copyright @ John Lipor, 2018
%--------------------------------------------------------------------------

    N = size(X, 2);
    
    if nargin < 6 || length(q) == 0
        q = N;
    end
    
    if nargin < 7 || length(T) == 0
        T = 30;
    end
    
    if nargin < 8 || length(heur) == 0
        heur = 0;
    end
    
    if nargin < 9 || length(sc) == 0
        sc = 1;
    end
    
    if nargin < 10 || length(base) == 0
        base = 'kss';
    end
    
    % form affinity matrix
    A = zeros(N);
    for bb = 1:B
        fprintf('repeat num: %d\n', bb);
        if strcmp(base,'kss')
            % [trueErr,l2Err,estLabels,state] = KSS(X,K,d,trueLabels,[],T);
            [~, est_labels] = KSS(X, K, true_labels, d, opts);
        else
            [trueErr,l2Err,est_labels,state] = copKSS(X,K,d,trueLabels,[],T,2);
        end
        for kk = 1:K
            kInds = find(est_labels==kk);
            if heur == 0
                A(kInds,kInds) = A(kInds,kInds) + 1;
            else
                A(kInds,kInds) = A(kInds,kInds) + 1 - l2Err;
            end
        end
    end
    A = A/B;
    
    % threshold
    A = A - diag(diag(A));
    if q < N
        Aq = thresh(A,q);
    else
        Aq = A;
    end
    
    % estimate labels using spectral clustering
    if sc == 1
        est_labels = SpectralClustering(Aq,K);
    else 
        est_labels = nan;
    end


end