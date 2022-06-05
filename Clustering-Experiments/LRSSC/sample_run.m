%% This script is a sample of running LRSSC
% mininmize_{C,E} |C|_* + lambda_1|C|_1+lambda_2/2|X-XC|^2_2
% subject to: diag(C)=0

% you will have a couple of parameters to tune to balance 
% the primal and dual residual to make it converge faster
% Check Boyd's paper fora guide line

clear all
clc
%% randomly generate data

n=50; k=6; r=4; kappa=5; m=r*kappa; N=m*k;
%generate ground truth
blk_mask=kron(eye(k),true(m));
blk_mask=logical(blk_mask);

Y=zeros(n,N);
S_base = cell(k,1);

cids=[];
for i=1:k
[U S V]=svds(randn(n,r),r);
S_base{i}=U;
V=randn(m,r);
Y(:,(i-1)*m+1:i*m)=U*V';
cids = [cids,i*ones(1,m)];
end

sigma=0.1/sqrt(n);

Z=randn(size(Y))*sigma;
Y=Y+Z;

for i=1:N
    Y(:,i)=Y(:,i)/norm(Y(:,i));
end

X=Y;



%% run  NoisySSC for comparison


lambda=(sigma*log(N)*sqrt(r)/log(kappa));
if lambda==0
    lambda=1/sqrt(n); % because anything greater than 1/r will work
end
lambda=1/lambda;
lambda=100;
%lambda=1/sqrt(n);
%lambda=sigma*log(N)*sqrt(r)/log(kappa);

lambda=sqrt(2*log(N))/(sigma*sqrt(n));
%by inexact ALM
tic;
[ C1, history ] = ALM_noisySSC( X,lambda);
% or the column by column version below
%[ C1, history ] = lassoSSC( X,1/lambda, 0.1*lambda, 1.5);
toc
ObjVal=history(end)/lambda
figure; imagesc(abs(C1));
set(gca,'position',[0 0 1 1]);
colormap(gray)
L=abs(C1)+abs(C1');
RelViolation=sum(L(~blk_mask))/sum(L(blk_mask))
gini=ginicoeff(L(blk_mask))

%% run ADMM NoisyLRSSC

lambda=0.1;

tic;
[C, history] = ALM_noisyLRSSC(X, lambda, sigma*sqrt(n), 0);
toc
ObjVal = history(end)
figure; imagesc(abs(C));
set(gca,'position',[0 0 1 1]);
colormap(gray)
L=abs(C)+abs(C');
RelViolation=sum(L(~blk_mask))/sum(L(blk_mask))
gini=ginicoeff(L(blk_mask))


% Note that the intra-class connections from LRSSC is much denser.



