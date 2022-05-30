% RISRO for robust PCA--Fully observed setting;
% Input: Y: observed matrix
% hatX: initialization; X: underlying parameter matrix;
% q: sparsity level of the gross error; r: input rank; p1, p2: matrix dimension
% iter_max: the maximum number of iterations
% Here the threshold.m function is from the code of paper "Robust PCA by Manifold Optimization" by Teng Zhang and Yi Yang.
function [error_matrix, succ_tag] = RISRO_robust_PCA(Y,hatX,X, q, r, p1, p2, iter_max,tol, succ_tol)
    [U0,~,V0] = svds(hatX,r);
    Ut = U0;
    Ut_perp = null(Ut');
    Vt = V0;
    Vt_perp = null(Vt');
    rela_err = norm(hatX - X, 'fro')/norm(X, 'fro');
    sparsity = ones(p1,p2);
    [~, select_index] = threshold(Y - hatX, 2 * q, sparsity);
    [y_grid,x_grid] = meshgrid(1:p2, 1:p1);
    newx_grid = x_grid.*select_index;
    newy_grid = y_grid.*select_index;
    newx_grid = newx_grid(:);
    newy_grid = newy_grid(:);
    mygrid = horzcat(newx_grid( newx_grid > 0 ),newy_grid( newy_grid > 0 ));
    error_matrix = [0, rela_err, 0];
    tic;
    for i = 1:iter_max
        n = size(mygrid,1);
        y = Y.*select_index;
        y = y(:);
        y = y(y~=0);
        tildeA = zeros(n, (p1+p2-r)*r);
        for j = 1:size(mygrid,1)
            Ab = kron(Vt(mygrid(j,2),:), Ut(mygrid(j,1) ,:));
            Ad1 = kron( Vt(mygrid(j,2),:), Ut_perp(mygrid(j,1) ,:)  );
            Ad2 = kron( Vt_perp(mygrid(j,2),:), Ut(mygrid(j,1) ,:) );
            tildeA(j,:) = [Ab, Ad1, Ad2];
        end
        gamma = (tildeA' * tildeA) \ tildeA' * y;
        %[gamma, flag] = lsqr(tildeA, y);
        hatB = reshape(gamma(1:r*r),[r,r]);
        hatD1 = reshape(gamma(r^2+1: r^2+r*(p1-r) ), [p1-r, r] );
        hatD2 = reshape(gamma( r^2+r*(p1-r)+1 : (p1+p2-r)*r ), [r, p2-r]);
        hatD2 = hatD2';
        % Newton step on the manifold
        hatXu = Ut * hatB + Ut_perp* hatD1;
        hatXv = Vt * hatB' + Vt_perp * hatD2;
        % retraction step
        hatX1 = hatXu * inv(hatB) * hatXv';
        [Ut,~] = qr(hatXu,0);
        [Vt,~] = qr(hatXv,0);
        Ut_perp = null(Ut');
        Vt_perp = null(Vt');
        rela_err = norm(hatX1 - X, 'fro')/norm(X, 'fro');
        [~, select_index] = threshold(Y - hatX1, 2 * q, sparsity);
        newx_grid = x_grid.*select_index;
        newy_grid = y_grid.*select_index;
        newx_grid = newx_grid(:);
        newy_grid = newy_grid(:);
        mygrid = horzcat(newx_grid( newx_grid > 0 ),newy_grid( newy_grid > 0 ));
        time = toc;
        iter_result = [i,rela_err, time];
        error_matrix = vertcat(error_matrix, iter_result);
        if rela_err < tol || rela_err > 5
            break
        end
    end
    if rela_err < succ_tol
        succ_tag = 1;
    else
        succ_tag = 0;
    end
end
