
clear all; clc;

%% parameters setting
n = 200;        %%% dimension of samples
K = 10;         %%% number of subspaces
N = K*400;      %%% number of samples
y = zeros(N,1);

%% dimensions of K subspaces 
d = 41; d_size = d-K:d-1; % d_size = randi([d-5,d-1],K,1);
d_max = max(d_size);

%% generate subspace bases with s intersections
s = 10; Y = randn(n,s); Us = Y*(Y'*Y)^(-0.5); 
for k = 1:K
    d1 = d_size(k); 
    Y = randn(n,d1-s); U0 = Y*(Y'*Y)^(-0.5);
    U1 = [Us U0];
    [Ut{k},~,~] = svd(U1,'econ');
end

V = Ut{1}'*Ut{2}; [U1,S1,V1] = svd(V);
aff = norm(S1,'fro')

%% generate samples in each subspace
Z = zeros(n,N);  %%% data matrix
Ht = zeros(N,K); %%% ground truth matrix
label_points = zeros(N,1);% label_points = randi([1 K], N, 1);
for k = 1:K
    inx = (k-1)*N/K+1:k*N/K;
    label_points(inx) = k;
end

for i = 1:N
    point_size = size(Ut{label_points(i)},2);
    b = randn(point_size,1);
    Z(:,i) = Ut{label_points(i)}*b/norm(b);
    Ht(i,label_points(i)) = 1;
end

%% Choose the running methods
run_KS = 1; run_TS = 0; run_GS = 0; run_LA = 0;

%% set the parameters
s = 1; iternum = 1e3; tol = 1e-8; print = 1; 

for iter = 1:s
    
%     fprintf('iter num: %d \n', iter);

    %% 
    if run_LA == 1
        tic; H_LA = lasso_SSC(Z); time = toc;
        mis_points_KS = misclassify_points(H_LA, Ht);
        fprintf('Robust SSC: wrong points = %d, time = %f\n', mis_points_KS, time_KS);
    end
    
    %% K-subspaces method for non-convex subspace clustering
    if run_KS == 1
        tau = 0.5*sqrt(log(N))/sqrt(d_max); %% threshold parameter
        init = 1; %% 1 = random initialization; 0 = initialization by TIP
        opts = struct('tau', tau, 'iternum', iternum, 'tol', tol, 'print', print, 'init', init);
        tic; [U_KS, H_KS, dist_collect_KS, misclass_collect_KS] = KSS(Z, K, Ut, Ht, d, opts); time_KS = toc;
        [mis_points_KS, dist_bases_KS] = dists_H_U(H_KS, U_KS, Ht, Ut);
        fprintf('K-subspace: wrong points = %d, basis error = %f, time = %f\n', mis_points_KS, dist_bases_KS, time_KS);
    end
    
    if run_TS == 1
        q = 12; %% the number of the nearest neighbors
        tic; [H_TS, U_TS] = TSC(Z, d, q); time_TS = toc;
        dist_bases_TS = norm(dist_subspace(U_TS, Ut, K));
        mis_points_TS = misclassify_points(H_TS, Ht);
        fprintf('TSC: wrong points = %d, basis error = %f, time = %f\n', mis_points_TS, dist_bases_TS, time_TS);
    end
    
    if run_GS == 1
        q = 30; %% the number of the nearest neighbors
        tic; [H_GS, U_GS] = GSC(Z, K, d, q); time_GS = toc;
        dist_bases_GS = norm(dist_subspace(U_GS, Ut, K));
        mis_points_GS = misclassify_points(H_GS, Ht);
        fprintf('TSC: wrong points = %d, basis error = %f, time = %f\n', mis_points_GS, dist_bases_GS, time_GS);
    end
    
    
end

%% plot the figures
if run_KS == 1
    semilogy(misclass_collect_KS+1e-8, '-s', 'Linewidth', 2); hold on;
    xlabel('Iterations', 'FontSize', 12); 
    ylabel('$\mathrm{d_F}(\mathbf{H},\mathbf{H}^*)$', 'Interpreter', 'latex', 'FontSize', 12);
end

