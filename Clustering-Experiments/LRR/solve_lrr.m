function e = solve_lrr(X, A, K, lambda, reg, alm_type, display, SC_type)
% Aug 2013
% This routine solves the following nuclear-norm optimization problem,
% min |Z|_*+lambda*|E|_L
% s.t., X = AZ+E
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
%        A -- D*M matrix of a dictionary, M is the size of the dictionary
%        lambda -- parameter
%        reg -- the norm chosen for characterizing E, 
%            -- reg=0 (default),                  use the l21-norm 
%            -- reg=1 (or ther values except 0),  use the l1-norm
%        alm_type -- 0 (default)   use the exact ALM strategy
%                 -- 1             use the inexact ALM strategy
%               
if nargin < 7 || isempty(display)
    display = false;
end
if nargin<6 || isempty(alm_type)
    alm_type = 0 ;
end

if nargin<5 || isempty(reg)
    reg = 0;
end

Q = orth(full(A'));
B = A*Q;

if reg==0
    if alm_type == 0 %% 0: exact ALM
        [Z,E] = exact_alm_lrr_l21v2(X,B,lambda,[],[],display);
    else %% 1: inexact ALM
        [Z,E] = inexact_alm_lrr_l21(X,B,lambda,display);
    end
else
    if alm_type == 0
        [Z,E] = exact_alm_lrr_l1v2(X,B,lambda,[],[],display);
    else
        [Z,E] = inexact_alm_lrr_l1(X,B,lambda,display);
    end
end

Z = Q*Z;
[U,S,~] = svd(Z, 'econ');
U1 = U*sqrt(S);
W = (U1*U1'); W = W.*W;
CKSym = BuildAdjacency(thrC(W,1));

if SC_type == 0
   e = SpectralClustering(CKSym, K);
else
   e = SpectralClustering_OMP(CKSym, K, 'Eig_Solver', 'eigs');
end


