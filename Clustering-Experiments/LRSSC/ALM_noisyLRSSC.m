
function est_labels = ALM_noisyLRSSC(Y, K, lambda, sigma, ADAPTIVE_ON, SC_type)
%ALM_LowRankSSC Summary of this function goes here
%   Detailed explanation goes here
% sigma is the estimated noise level
% lambda defines the trade-off between nuclear norm and one-norm

% it solves the following problem
%-----------------------------------------
% min 0.5||X-XC||^2_F + lambda1||C||_* + lambda2||C||_1
% subject to: diag(C)=0
%-----------------------------------------
% lambda1=2*sigma*beta/n
% lambda2=2*sigma*(1-beta)/n
tic; 
[m,n]=size(Y);
t=1+lambda;
beta=1/t;
beta2=1-beta;

lambda1=beta/sqrt(2*log(n))*sigma;
lambda2=beta2/sqrt(2*log(n))*sigma;



mu=0.2;

mu2=mu;
mu3=mu;
% the ratio between mu1, mu2, mu3 can be tuned
% but once these are changed once, one matrix inversion need to be computed


% optimal variables
rho0 = 1.1;% this one can be tuned
rho=1;

epsilon=0.1;
eta=epsilon*5*n;
max_mu=1000;

tol_abs=1e-5;
tol_rel=1e-3;

max_iter = 2e2;
iter=0;

%initialze
C1=zeros(n,n);
C2=zeros(n,n);
J=zeros(n,n);
Lambda2=zeros(n,n);
Lambda3=zeros(n,n);
fun_val=0;


%caching a few quantities
YTY=(Y'*Y); 
invMat=pinv(YTY+(mu2+mu3)*eye(n));


diag_idx=logical(eye(n));
converge=0;
while ~converge
    if rho~=1
        invMat=inv(YTY+(mu2+mu3)*eye(n));
    end
    

    J_new=invMat*(YTY+mu2*C2+mu3*C1-Lambda2-Lambda3);
    %J_new=invMat*(mu1_YTY+mu2*C2+Y'*Lambda1-Lambda2);
    C2_new=soft_thresh(J_new+Lambda2/mu2,lambda2/mu2);
    C2_new(diag_idx)=0;
    C1_new=sigma_soft_thresh(J_new+Lambda3/mu3,lambda1/mu3);
    
    
    %primal and dual residuals
    d_res=mu2*norm(C1_new(:)-C1(:))+mu3*norm(C2_new(:)-C2(:));
    pDist2=J_new-C2_new;
    pDist3=J_new-C1_new;
    p_res=norm(pDist2(:))+norm(pDist3(:));
    
    %update
    C1=C1_new;
    C2=C2_new;
    J=J_new;     
    Lambda2=Lambda2+mu2*pDist2;
    Lambda3=Lambda3+mu3*pDist3;
    
    if ADAPTIVE_ON
        rho=min(max_mu/mu2,rho0);
        if d_res*eta/norm(Y(:))>epsilon
            rho=1;
        else            
            mu2=rho*mu2;
            mu3=rho*mu3;
        end
    end
    
    
   % It is pointless to know function value
   % cur_fun_val=lambda1*sum(svd(C1))+lambda2*norm(C1(:),1);
   % res_func=abs(cur_fun_val-fun_val); 
     
   iter=iter+1;
%    if mod(iter,20)==0
%         fprintf('%d th iteration compelte, P_res= %f, D_res=%f\n',iter,p_res,d_res);
% %     fprintf('    Feasibility is :[%f, %f] mu=[%.2f, %.2f]\n',norm(pDist2(:)),norm(pDist3(:)),...
% %         mu2,mu3);
%    end
   history(iter)=p_res;
   
   
   if( p_res<n*tol_abs+tol_rel*max(norm(C1,'fro'),norm(J,'fro')) && d_res <n*tol_abs+tol_rel*norm([Lambda2(:);Lambda3(:)]))
%   if( p_res<n*tol_abs && d_res <n*tol_abs)
       converge=1;
       fprintf('Converged.\n')
   end
   if iter>=max_iter 
   converge=1;
   fprintf('Maximum iteration reached. Quit.\n')
   end
   
   if toc > 1800
       break;
   end
   %fun_val=cur_fun_val;
end

CKSym = BuildAdjacency(thrC(C1,0.8));
if SC_type == 0
   est_labels = SpectralClustering(CKSym, K);
else
   est_labels = SpectralClustering_OMP(CKSym, K, 'Eig_Solver', 'eigs');
end
