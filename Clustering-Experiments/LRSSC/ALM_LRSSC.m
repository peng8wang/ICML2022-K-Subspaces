function [ C1, history ] = ALM_LRSSC( Y,lambda, ADAPTIVE_ON)
%ALM_LowRankSSC Summary of this function goes here
%   Detailed explanation goes here

t=1+lambda;
lambda1=1/t;
lambda2=lambda/t;

[m,n]=size(Y);

mu=0.1;
mu1=10*mu;% this one can be tuned
mu2=10*mu;
mu3=mu;

% optimal variables
rho0 = 1.1;% this one can be tuned
rho=1;

epsilon=0.1;
eta=epsilon*5*sqrt(m*n);
max_mu=10000;

tol_abs=1e-5;
tol_rel=1e-3;

max_iter=500;
iter=0;

%initialze
C1=zeros(n,n);
C2=zeros(n,n);
J=zeros(n,n);
Lambda1=zeros(m,n);
Lambda2=zeros(n,n);
Lambda3=zeros(n,n);
fun_val=0;


%caching a few quantities
mu1_YTY=mu1*(Y'*Y); 
invMat=inv(mu1_YTY+(mu2+mu3)*eye(n));


diag_idx=logical(eye(n));
converge=0;
while ~converge

    if rho~=1
        invMat=1/rho*invMat;
        mu1_YTY=rho*mu1_YTY;
    end
    

    J_new=invMat*(mu1_YTY+mu2*C2+mu3*C1+Y'*Lambda1-Lambda2-Lambda3);
    %J_new=invMat*(mu1_YTY+mu2*C2+Y'*Lambda1-Lambda2);
    C2_new=soft_thresh(J_new+Lambda2/mu2,lambda2/mu2);
    C2_new(diag_idx)=0;
    C1_new=sigma_soft_thresh(J_new+Lambda3/mu3,lambda1/mu3);
    
    
    %primal and dual residuals
    d_res=norm(C1_new(:)-C1(:))+norm(C2_new(:)-C2(:));
    pDist1=Y-Y*J_new;
    pDist2=J_new-C2_new;
    pDist3=J_new-C1_new;
    p_res=norm(pDist1(:))+norm(pDist2(:))+norm(pDist3(:));
    
    %update
    C1=C1_new;
    C2=C2_new;
    J=J_new;     
    Lambda1=Lambda1+mu1*pDist1;
    Lambda2=Lambda2+mu2*pDist2;
    Lambda3=Lambda3+mu3*pDist3;
    
    if ADAPTIVE_ON
        rho=min(max_mu/mu1,rho0);
        if d_res*eta/norm(Y(:))>epsilon
            rho=1;
        else            
            mu1=rho*mu1;
            mu2=rho*mu2;
            mu3=rho*mu3;
        end
    end

   
   cur_fun_val=lambda1*sum(svd(C1))+lambda2*norm(C1(:),1);   
   % res_func=abs(cur_fun_val-fun_val); 
     
   iter=iter+1;
   if mod(iter,20)==0
    fprintf('%d th iteration compelte, P_res= %f, D_res=%f, Obj = %f\n',iter,p_res,d_res,cur_fun_val);
    fprintf('    Feasibility is :[%f, %f, %f] mu=[%.2f, %.2f, %.2f]\n',norm(pDist1(:)),norm(pDist2(:)),norm(pDist3(:)),...
        mu1,mu2,mu3);
   end
   history(iter)=cur_fun_val;
   
   
   if( p_res<n*tol_abs+tol_rel*max(norm(C1,'fro'),norm(J,'fro')) && d_res <n*tol_abs+tol_rel*norm([Lambda1(:);Lambda2(:);Lambda3(:)]))
%   if( p_res<n*tol_abs && d_res <n*tol_abs)
       converge=1;
       fprintf('Converged.\n')
   end
   if iter>=max_iter
   converge=1;
   fprintf('Maximum iteration reached. Quit.\n')
   end

   
   %fun_val=cur_fun_val;
end
end