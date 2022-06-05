
%% Tests on real-world data sets
clear; clc;

%% load the folders of the tested methods
addpath('./ADMM_SSC'); addpath('./LRR');
addpath('./LRSSC'); addpath('./SSC-OMP')

%% load and pre-process the data sets 
load('./datasets/Ex_YaleB.mat')
Z = fea'; true_labels = gnd; %%% Z: data matrix; labels: ground truth;
if min(gnd) == 0
    true_labels = gnd + 1;
end
Z = double(Z); Z = normc(Z);

%% parameter settings
K = max(gnd); %%% number of subspaces
[n, N] = size(Z); %%% N: number of samples
d = 8; %%% dimension of subspaces

%% choose the running algorithms
run_KS = 1; run_SC = 0; run_TS = 0; run_GS = 0; run_LR = 0; run_LC = 0; run_MP = 0;

%% run the algorithms
repeat_num = 10; max_iternum = 2e2;

[acc_KS_collect, acc_SC_collect, acc_LR_collect,...
 acc_LC_collect, acc_TS_collect, acc_GS_collect, acc_MP_collect] = deal(zeros(repeat_num,1));
[ttime_KS, ttime_SC, ttime_LR, ttime_LC, ttime_TS, ttime_GS, ttime_MP] = deal(0);

for repeat = 1:repeat_num

    fprintf('Number of test: %d \n', repeat);

    %% K-Subspaces(KSS) method
    if run_KS == 1
        tau = 9.8e-1; SC_type = 0; %% 0 = normalized spectral clustering
        init = 0; %% 1 = random initialization; 0 = initialization by TIPS
        opts = struct('tau', tau, 'iternum', max_iternum, 'tol', 1e-2, 'print', 0, 'init', init, 'SC_type', SC_type);
        tic; [U_KS, e_KS] = KSS(Z, K, true_labels, d, opts); time = toc;
        acc = 1 - misRate(true_labels, e_KS)/N;
        acc_KS_collect(repeat) = acc; ttime_KS = ttime_KS + time;
        fprintf('KSS: accuracy = %.4f, time = %f\n', acc, time);
    end
    
    %% ADMM for Sparse Subspace Clustering (SSC)
    if run_SC == 1
        [D,N] = size(Z); alpha = 10;      
        r = 0; affine = false; outlier = true; rho = 1;
        tic; e_SC = SSC(Z, r, affine, alpha, outlier, rho, true_labels, SC_type); time = toc;
        acc = 1 - misRate(true_labels, e_SC)/N;
        acc_SC_collect(repeat) = acc; ttime_SC = ttime_SC + time;
        fprintf('ADMM SSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% Thresholding-based Subspace Clustering (TSC)
    if run_TS == 1
        q = 4;
        tic; e_TS = TSC(Z, d, K, SC_type, q); time = toc;
        acc = 1 - misRate(true_labels, e_TS)/N;
        acc_TS_collect(repeat) = acc; ttime_TS = ttime_TS + time;
        fprintf('TSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% Greedy Subspace Clustering (GSC) 
    if run_GS == 1
        q = 20;
        tic; e_GS = GSC(Z, K, d, SC_type, q); time = toc;
        acc = 1 - misRate(true_labels, e_GS)/N;
        acc_GS_collect(repeat) = acc; ttime_GS = ttime_GS + time;
        fprintf('GSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% ALM for Low-Rank Representation (LRR) for Subspace Clustering
    if run_LR == 1
        lambda = 1e-1; reg = 1;  
        tic; e_LR = solve_lrr(Z, Z, K, lambda, reg, 1, false, SC_type); time = toc;
        acc = 1 - misRate(true_labels, e_LR)/N; 
        acc_LR_collect(repeat) = acc; ttime_LR = ttime_LR + time;
        fprintf('ALM LRR: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% ADMM for Low-Rank Sparse Subspace Clustering (LRR-SSC)
    if run_LC == 1
        sigma = 0.1; lambda = 1; 
        tic; e_LC = ALM_noisyLRSSC(Z, K, lambda, sigma, 1, SC_type); time = toc;
        acc = 1 - misRate(true_labels, e_LC)/N;
        acc_LC_collect(repeat) = acc; ttime_LC = ttime_LC + time;
        fprintf('ADMM LRR-SSC: accuracy = %.4f, time = %f\n', acc, time);
    end

    %% Sparse Subspace Clustering with Orthogonal Matching Pursuit
    if run_MP == 1
        tic; e_MP = SSC_OMP(Z, N, 5, K, 0); time = toc;
        acc = 1 - misRate(true_labels, e_MP)/N;
        acc_MP_collect(repeat) = acc; ttime_MP = ttime_MP + time;
        fprintf('SSC-OMP: accuracy = %.4f, time = %f\n', acc, time);
    end
end

if run_KS*run_SC*run_GS*run_TS*run_LR*run_LC == 1

    %% maximum clustering accuracy
    acc_KS = max(acc_KS_collect); acc_SC = max(acc_SC_collect);
    acc_LR = max(acc_LR_collect); acc_LC = max(acc_LC_collect);
    acc_TS = max(acc_TS_collect); acc_GS = max(acc_GS_collect);
    acc_MP = max(acc_MP_collect);
    fprintf(['Max accuracy of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f\n'], acc_KS, acc_SC, acc_TS, acc_GS, acc_LR, acc_LC, acc_MP);

    %% The 2nd largest clustering accuracy
    acc_KS = maxk(acc_KS_collect, 2); acc_SC = maxk(acc_SC_collect, 2);
    acc_LR = maxk(acc_LR_collect, 2); acc_LC = maxk(acc_LC_collect, 2);
    acc_TS = maxk(acc_TS_collect, 2); acc_GS = maxk(acc_GS_collect, 2);
    acc_MP = maxk(acc_MP_collect, 2);
    fprintf(['2nd max accuracy of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f\n'], acc_KS(2), acc_SC(2), acc_TS(2), acc_GS(2), acc_LR(2), acc_LC(2), acc_MP(2));
    
    %% average accuracy
    acc_KS = mean(acc_KS_collect); acc_SC = mean(acc_SC_collect);
    acc_LR = mean(acc_LR_collect); acc_LC = mean(acc_LC_collect);
    acc_TS = mean(acc_TS_collect); acc_GS = mean(acc_GS_collect);
    acc_MP = mean(acc_MP_collect);
    fprintf(['Average accuracy of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f\n'], acc_KS, acc_SC, acc_TS, acc_GS, acc_LR, acc_LC, acc_MP);

    %% average running time
    time_KS = ttime_KS/repeat_num; time_SC = ttime_SC/repeat_num; 
    time_LR = ttime_LR/repeat_num; time_LC = ttime_LC/repeat_num;
    time_TS = ttime_TS/repeat_num; time_GS = ttime_GS/repeat_num;  
    time_MP = ttime_MP/repeat_num;
    fprintf(['CPU time of KS: %.4f, SSC: %.4f, TSC: %.4f, GSC: %.4f, ' ...
        'LRR: %.4f, LRR-SSC: %.4f, OMP: %.4f \n'], time_KS, time_SC, time_TS, time_GS, time_LR, time_LC, time_MP);
end
save('result-YaeB-new')