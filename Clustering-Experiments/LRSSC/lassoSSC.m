function [C, ObjVal]=lassoSSC(X, lambda, rho,alpha)

[n,N]=size(X);
ObjVal=0;
C=zeros(N);
for i=1:N
x=X(:,i);
if i==1
idx=2:N;
elseif i==N
idx=1:N-1;  
else
idx=[1:i-1,i+1:N];
end

%lambda=1/sqrt(n);

%by ADMM (fast)
[c history] = lasso_ADMM(X(:,idx), x, lambda, rho, alpha); ObjVal=ObjVal+history.objval(end);

C(idx,i)=c;
        if mod(i,100)==0
            fprintf('%d th column done.\n',i);
        end
end

end