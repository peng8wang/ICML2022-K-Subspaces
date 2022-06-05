
clear; clc;

%% parameters setting
n = 300;        %%% dimension of samples
K = 9;          %%% number of subspaces
Nk = 500;       %%% number of samples in each subspace
N = K*Nk;       %%% total number of samples

%% dimensions of K subspaces 
d_min = 25; d_max = 30;
d_size = randi([d_min,d_max],K,1); d = max(d_size) + 1;

%% generate subspace bases with s intersections
s = 6; Y = rand(n); [U,~,~] = svd(Y); % U = Y*(Y'*Y)^(-0.5); 
Us = U(:,n-s+1:n);

for k = 1:K
    d1 = d_size(k);
    I = randperm(n-s); inx = I(1:d1-s);
    Ut{k} = [U(:,inx) Us]; 
end

%% generate ground truth 
Z = zeros(n,N);  %%% data matrix
true_labels = zeros(N,1); %%% true labels of data points
for k = 1:K
    inx = (k-1)*N/K+1:k*N/K; true_labels(inx) = k;
end

%% set the parameters
run_KS = 1; run_times = 3; iternum = 15; tol = 1e-8; print = 1; max_iter = 1;

for iter = 1:run_times

    %% generate samples in each subspace
    for i = 1:N
        point_size = size(Ut{true_labels(i)},2);
        b = randn(point_size,1);
        Z(:,i) = Ut{true_labels(i)}*b/norm(b);
    end

    fprintf('iter num: %d \n', iter);
    if iter == 1
        color = '#0072BD';
    elseif iter == 2 
        color = '#D95319';
    elseif iter == 3 
        color = '#7E2F8E';
    elseif iter == 4
        color = '#77AC30';
    else
        color = '#A2142F';
    end
    
    %% K-subspaces method for non-convex subspace clustering
    if run_KS == 1
        tau = 2/sqrt(d_max); %% threshold parameter
        init = 1; %% 1 = random initialization; 0 = initialization by TIP
        opts = struct('tau', tau, 'iternum', iternum, 'tol', tol, 'print', print, 'init', init);
        tic; [U_KS1, e_KS1, misclass_collect_KS1] = KSS(Z, K, true_labels, d, opts); time_KS = toc;
        mis_points_KS = misRate(true_labels, e_KS1); max_iter = max(max_iter, size(misclass_collect_KS1,2));
        fprintf('K-subspace: wrong points = %d, time = %f\n', mis_points_KS, time_KS);
    end

    if run_KS == 1
        tau = 2/sqrt(d_max); 
        init = 0; %% 1 = random initialization; 0 = initialization by TIPS
        opts = struct('tau', tau, 'iternum', iternum, 'tol', tol, 'print', print, 'init', init);
        tic; [U_KS2, e_KS2, misclass_collect_KS2] = KSS(Z, K, true_labels, d, opts); time_KS = toc;
        mis_points_KS = misRate(true_labels, e_KS2);
        fprintf('K-subspace: wrong points = %d, time = %f\n', mis_points_KS, time_KS);
    end
    
    %% plot the figures
    semilogy(misclass_collect_KS1+1e-8, '-s', 'Color', color, 'DisplayName',...
        ['RI-KSS on data ' num2str(iter) ''], 'LineWidth', 2.5, 'MarkerSize', 6); hold on;
    semilogy(misclass_collect_KS2+1e-8, '-o', 'Color', color, 'DisplayName',...
        ['TI-KSS on data ' num2str(iter) ''], 'LineWidth', 2.5, 'MarkerSize', 6);        

end

legend('show'); xlim([1, max_iter+4]);
xlabel('$\textbf{Iterations}$', 'Interpreter', 'latex', 'FontSize', 12); 
ylabel('$\mathbf{d_F^2}(\mathbf{H},\mathbf{H}^*)$', 'Interpreter', 'latex', 'FontSize', 12);


