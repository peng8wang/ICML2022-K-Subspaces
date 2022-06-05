function [ C, history ] = ALM_noisySSC( Y,lambda )
%ALM_ROBUSTSSC Summary of this function goes here
%   Detailed explanation goes here
[m,n]=size(Y);

mu=0.2*lambda;% this one can be tuned
rho = 1.0;% this one can be tuned
max_mu=1e10;

tol_abs=1e-4;
tol_rel=1e-3;

max_iter=500;
iter=0;

%initialze
C=zeros(n,n);
J=zeros(n,n);
E=zeros(m,n);
Lambda=zeros(n,n);
fun_val=0;


%caching a few quantities
lambda_YTY=lambda*(Y'*Y); 
invMat=inv(lambda_YTY+mu*eye(n));


diag_idx=logical(eye(n));
converge=0;
while ~converge

    if rho~=1
        invMat=pinv(lambda_YTY+mu*speye(n));
    end
    

    J_new=invMat*(lambda_YTY+mu*C-Lambda);
    %J=(lambda*YTY+mu*C-Lambda)\(mu1*YTY+mu2);
    C_new=soft_thresh(J_new+Lambda/mu,1/mu);
    C_new(diag_idx)=0;
    
    
    %primal and dual residuals
    d_res=mu*norm(C_new(:)-C(:));
    pDist=J_new-C_new;
    p_res=norm(pDist(:));
    
    %update
    C=C_new;
    J=J_new;     
    Lambda=Lambda+mu*pDist;    
    mu = min(max_mu,mu*rho);    

   
   cur_fun_val=norm(C(:),1)+ lambda/2*norm(Y-Y*C,'fro')^2;   
   % res_func=abs(cur_fun_val-fun_val); 
     
   iter=iter+1;
   if mod(iter,20)==0
   fprintf('%d th iteration compelte, P_res= %f, D_res=%f, Obj = %f\n',iter,p_res,d_res,cur_fun_val);
   end
   history(iter)=cur_fun_val;
   
   
   if( p_res<n*tol_abs+tol_rel*max(norm(C,'fro'),norm(J,'fro')) && d_res <n*tol_abs+tol_rel*norm(Lambda,'fro'))
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