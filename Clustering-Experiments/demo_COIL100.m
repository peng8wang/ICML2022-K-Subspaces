
%% Tests on real-world data sets
clear; clc;

%% load the folders of the tested methods
addpath('./ADMM_SSC'); addpath('./LRR');
addpath('./LRSSC'); addpath('./SSC-OMP')

%% load and pre-process the data sets 

load('./datasets/COIL100.mat')
Z = fea'; true_labels = gnd; %%% Z: data matrix; labels: ground truth;
% load('.\datasets\coil100Reduced.mat'); %%% load pre-processed data
% Z = X; true_labels = trueLabels;
if min(true_labels) == 0
    true_labels = true_labels + 1;
end
Z = double(Z); Z = normc(Z);

%% parameter settings
K = max(true_labels); %%% number of subspaces
[n, N] = size(Z); %%% N: number of samples
d = 10; %%% dimension of subspaces

%% choose the running algorithms
run_KS = 1; run_SC = 1; run_TS = 1; run_GS = 1; run_LR = 1; run_LC = 1; run_MP = 1;

%% run the algorithms
repeat_num = 1; max_iternum = 2e2;

[acc_KS_collect, acc_SC_collect, acc_LR_collect,...
 acc_LC_collect, acc_TS_collect, acc_GS_collect, acc_MP_collect] = deal(zeros(repeat_num,1));
[time_KS_collect, time_SC_collect, time_LR_collect,...
 time_LC_collect, time_TS_collect, time_GS_collect, time_MP_collect] = deal(zeros(repeat_num,1));

for repeat = 1:repeat_num

    fprintf('Number of test: %d \n', repeat);

    %% K-Subspaces(KSS) method
    if run_KS == 1
        tau = 9.8e-1; SC_type = 1; %% 0 = normalized spectral clustering
        init = 0; %% 1 = random initialization; 0 = initialization by TIPS
        opts = struct('tau', tau, 'iternum', max_iternum, 'tol', 1e-2, 'print', 0, 'init', init, 'SC_type', SC_type);
        tic; [U_KS, e_KS] = KSS(Z, K, true_labels, d, opts); time = toc;
        acc = 1 - misRate(true_labels, e_KS)/N;
        acc_KS_collect(repeat) = acc; time_KS_collect(repeat) = time;
        fprintf('KSS: accuracy = %.4f, time = %f\n', acc, time);
    end
    
    %% ADMM for Sparse Subspace Clustering (SSC)
    if run_SC == 1
        [D,N] = size(Z); alpha = 10;  
        r = 0; affine = false; outlier = true; rho = 2;
        tic; e_SC = SSC(Z, r, affine, alpha, outlier, rho, true_labels, SC_type); time = toc;
        acc = 1 - misRate(true_labels, e_SC)/N;
        acc_SC_collect(repeat) = acc; time_SC_collect(repeat) = time;
        fprintf('ADMM SSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% Thresholding-based Subspace Clustering (TSC)
    if run_TS == 1
        q = 3;
        tic; e_TS = TSC(Z, d, K, SC_type, q); time = toc;
        acc = 1 - misRate(true_labels, e_TS)/N;
        acc_TS_collect(repeat) = acc; time_TS_collect(repeat) = time;
        fprintf('TSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% Greedy Subspace Clustering (GSC) 
    if run_GS == 1
        q = 15;
        tic; e_GS = GSC(Z, K, d, SC_type, q); time = toc;
        acc = 1 - misRate(true_labels, e_GS)/N;
        acc_GS_collect(repeat) = acc; time_GS_collect(repeat) = time;
        fprintf('GSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% ALM for Low-Rank Representation (LRR) for Subspace Clustering
    if run_LR == 1
        lambda = 1e-3; reg = 1; 
        tic; e_LR = solve_lrr(Z, Z, K, lambda, reg, 1, 0, SC_type); time = toc;
        acc = 1 - misRate(true_labels, e_LR)/N; 
        acc_LR_collect(repeat) = acc; time_LR_collect(repeat) = time;
        fprintf('ALM LRR: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% ADMM for Low-Rank Sparse Subspace Clustering (LRR-SSC)
    if run_LC == 1
        sigma = 1; lambda = 2; 
        tic; e_LC = ALM_noisyLRSSC(Z, K, lambda, sigma, 1, SC_type); time = toc;
        acc = 1 - misRate(true_labels, e_LC)/N;
        acc_LC_collect(repeat) = acc; time_LC_collect(repeat) = time;
        fprintf('ADMM LRR-SSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% Sparse Subspace Clustering with Orthogonal Matching Pursuit
    if run_MP == 1
        tic; e_MP = SSC_OMP(Z, N, 2, K, 0); time = toc;
        acc = 1 - misRate(true_labels, e_MP)/N;
        acc_MP_collect(repeat) = acc; time_MP_collect(repeat) = time;
        fprintf('SSC-OMP: accuracy = %.4f, time = %f\n', acc, time);
    end
end

if run_KS*run_SC*run_GS*run_TS*run_LR*run_LC == 0

    %% maximum clustering accuracy
    acc_KS = max(acc_KS_collect); acc_SC = max(acc_SC_collect);
    acc_LR = max(acc_LR_collect); acc_LC = max(acc_LC_collect);
    acc_TS = max(acc_TS_collect); acc_GS = max(acc_GS_collect);
    acc_MP = max(acc_MP_collect);
    fprintf(['Max accuracy of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f\n'], acc_KS, acc_SC, acc_TS, acc_GS, acc_LR, acc_LC, acc_MP);

    %% average accuracy
    acc_KS = mean(acc_KS_collect); acc_SC = mean(acc_SC_collect);
    acc_LR = mean(acc_LR_collect); acc_LC = mean(acc_LC_collect);
    acc_TS = mean(acc_TS_collect); acc_GS = mean(acc_GS_collect);
    acc_MP = mean(acc_MP_collect);
    fprintf(['average accuracy of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f\n'], acc_KS, acc_SC, acc_TS, acc_GS, acc_LR, acc_LC, acc_MP);

    %% standard deviation
    acc_KS = std(acc_KS_collect); acc_SC = std(acc_SC_collect);
    acc_LR = std(acc_LR_collect); acc_LC = std(acc_LC_collect);
    acc_TS = std(acc_TS_collect); acc_GS = std(acc_GS_collect);
    acc_MP = std(acc_MP_collect);
    fprintf(['std of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f\n'], acc_KS, acc_SC, acc_TS, acc_GS, acc_LR, acc_LC, acc_MP);
    
    %% average running time
    time_KS = mean(time_KS_collect); time_SC = mean(time_SC_collect); 
    time_LR = mean(time_LR_collect); time_LC = mean(time_LC_collect);
    time_TS = mean(time_TS_collect); time_GS = mean(time_GS_collect);  
    time_MP = mean(time_MP_collect);
    fprintf(['average time of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f \n'], time_KS, time_SC, time_TS, time_GS, time_LR, time_LC, time_MP);

    %% std of running time
    time_KS = std(time_KS_collect); time_SC = std(time_SC_collect); 
    time_LR = std(time_LR_collect); time_LC = std(time_LC_collect);
    time_TS = std(time_TS_collect); time_GS = std(time_GS_collect);  
    time_MP = std(time_MP_collect);
    fprintf(['std of time of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f \n'], time_KS, time_SC, time_TS, time_GS, time_LR, time_LC, time_MP);
end
save('Result-COIL100.mat')
