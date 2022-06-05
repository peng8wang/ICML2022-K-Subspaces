%--------------------------------------------------------------------------
% This is the function to call the sparse optimization program, to call the 
% spectral clustering algorithm and to compute the clustering error.
% r = projection dimension, if r = 0, then no projection
% affine = use the affine constraint if true
% s = clustering ground-truth
% missrate = clustering error
% CMat = coefficient matrix obtained by SSC
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [est_labels,CMat] = SSC(X,r,affine,alpha,outlier,rho,s,SC_type)

if (nargin < 6)
    rho = 1;
end
if (nargin < 5)
    outlier = false;
end
if (nargin < 4)
    alpha = 20;
end
if (nargin < 3)
    affine = false;
end
if (nargin < 2)
    r = 0;
end

n = max(s);
Xp = DataProjection(X,r);

if (~outlier)
    CMat = admmLasso_mat_func(Xp,affine,alpha,1e-3,2e2);
    C = CMat;
else
    CMat = admmOutlier_mat_func(Xp,affine,alpha,1e-3,2e2);
    N = size(Xp,2);
    C = CMat(1:N,:);
end

CKSym = BuildAdjacency(thrC(C,rho));
if SC_type == 0
    est_labels = SpectralClustering(CKSym,n);
else
    est_labels = SpectralClustering_OMP(CKSym, n, 'Eig_Solver', 'eigs');
end
% misclass_points = misRate(s, est_labels);

