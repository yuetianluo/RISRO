% This code is for RISRO on Phase Retrieval 
clear all
tic;
rng('default');
rng(2021);
p = 128;
n = 10*p;
sig = 0;
lambda = 3;

u = randn(p,1);
u =  u/norm(u,'fro');
x = lambda * u;

A = randn(n,p);
eps = sig * randn(n,1); 
y = zeros(n,1);
for i = 1:n
    y(i) = (x' * (A(i,:))')^2 + eps(i);
end

% good initialization
tildex = zeros(p,p);
for i = 1:n
    tildex = tildex + y(i) * (A(i,:))' * A(i,:);
end
tildex = tildex/n;
[u0,sigma0] = eigs(tildex,1);
x0 = u0 * sqrt(sigma0);
iter_max = 30;
tol = 1e-13;
succ_tol = 1e-13;
[error_matrix,~] = RISRO_Phase_Retrieval(A, y, p, x, x0, iter_max, tol,succ_tol);
array2table(error_matrix)
time = toc;
time



