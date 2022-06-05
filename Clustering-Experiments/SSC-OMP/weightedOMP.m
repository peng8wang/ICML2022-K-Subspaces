function C = weightedOMP(X, K, W, thr)

[D, N] = size(X);

Xn = X; % assume that data is column normalized. If not, uncomment the following.
Xn = cnormalize(X);

C = zeros(N);
res = Xn;
for ii = 1:N
    Si = find(W(ii,:)==1);
    t = length(Si);
    while t < K
        % get point with maximum inner product
        ip = abs(Xn'*res(:,ii));
        ip(ii) = 0;
        [~,ind] = max(ip,[],1);
        Si = [Si ind];
        % compute residual
        B = X(:,Si);
        res(:,ii) = Xn(:,ii) - B*(B\Xn(:,ii));
        t = length(Si);
    end
    C(ii,Si) = Xn(:,Si)\Xn(:,ii);
end
