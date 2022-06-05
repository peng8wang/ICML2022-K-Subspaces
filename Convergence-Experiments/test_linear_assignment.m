

N = 10000; K = 2; 

et = randi([1 K],N,1); Ht = zeros(N,K);
for k = 1:K
    Ht(et==k,k) = 1;
end

e = randi([1 K],N,1); H = zeros(N,K);
for k = 1:K
    H(e==k,k) = 1;
end

%% linear programming
tic; misclass_num = dists_H(H, Ht); time = toc; 

%% Hungarian algorithm
tic; misclass_num1 = missRate(et, e)*N; time1 = toc;

time/time1
