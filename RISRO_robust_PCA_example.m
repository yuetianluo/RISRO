% RISRO for robust PCA example. For simplicity, we consider the setting Y is fully observed.
clear all
tic;
rng('default');
rng(2021);
p1 = 50;
p2 = 50;
r = 3;
sig = 0;
lambda = 3;
iter_max = 30;
tol = 1e-13;
succ_tol = 1e-13;

S = diag(repelem(lambda,r));
U = randn(p1,r);
[U,~,~] = svds(U,p1);
V = randn(p2,r);
[V,~,~] = svds(V,p2);

X = U*S*(V');
X = X + sig * randn(p1, p2);

sparsity = ones(p1,p2);
% sparsity level of the gross error
q = 0.02;
A = binornd(1,q*ones(p1));
S = 10*randn(p1,p2);
S = S.* A;
Y = S + X;

threshold_ratio = 5;

[X0, select_index] = threshold(Y, threshold_ratio * q, sparsity);
[U0,Sigma0,V0] = svds(X0, r);
X_init = U0 * Sigma0 * V0';

 
[error_matrix,~] = RISRO_robust_PCA(Y, X_init, X, q, r, p1, p2, threshold_ratio, iter_max, tol, succ_tol);
array2table(error_matrix)
time = toc;
time

