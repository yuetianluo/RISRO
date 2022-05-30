clear all
tic;
rng('default');
rng(2021);
p1 = 100;
p2 = 100;
r = 3;
sig = 0;
lambda = 3;

over_sam = 5;
iter_max = 30;
tol = 1e-13;
succ_tol = 1e-13;

rho = min((p1+p2-r)*r/(p1*p2)*over_sam,0.9);
S = diag(repelem(lambda,r));
U = randn(p1,r);
[U,~,~] = svds(U,p1);
V = randn(p2,r);
[V,~,~] = svds(V,p2);

X = U*S*(V');
X = X + sig * randn(p1, p2);

[x_grid,y_grid] = meshgrid(1:p2, 1:p1);
mygrid = horzcat(y_grid(:),x_grid(:));
y = X(:);
index = randperm(p1*p2);
n = ceil(p1*p2*rho);
index_select = index(1:n);
index_nonselect = index((n+1):(p1*p2));
mygrid = mygrid(index_select,:);
y_select = y(index_select);

% good initialization


y(index_nonselect) = 0;
X_init = reshape(y, p1, p2);
[U0,Sigma0,V0] = svds(X_init, r);
hatX0 = U0 * Sigma0 * V0';
[error_matrix,~] = RISRO_matrix_completion(y_select,mygrid, hatX0,X, r, p1, p2, iter_max, tol, succ_tol);
array2table(error_matrix)
time = toc;
time
