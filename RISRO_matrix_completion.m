% RISRO for matrix completion;
% Input: y: observed entries with order determined by mygrid;
% mygrid: index pairs for observed entries; hatX: initialization; X: underlying parameter matrix; r: input rank; p1, p2: matrix dimension
% iter_max: the maximum number of iterations
function [error_matrix, succ_tag] = RISRO_matrix_comppletion(y,mygrid, hatX,X, r, p1, p2, iter_max,tol, succ_tol)
    [U0,~,V0] = svds(hatX,r);
    Ut = U0;
    Ut_perp = null(Ut');
    Vt = V0;
    Vt_perp = null(Vt');
    rela_err = norm(hatX - X, 'fro')/norm(X, 'fro');
    error_matrix = [0, rela_err, 0];
    n = length(y);
    tic;
    for i = 1:iter_max
        tildeA = zeros(n, (p1+p2-r)*r);
        for j = 1:size(mygrid,1)
            Ab = kron(Vt(mygrid(j,2),:), Ut(mygrid(j,1) ,:));
            Ad1 = kron( Vt(mygrid(j,2),:), Ut_perp(mygrid(j,1) ,:)  );
            Ad2 = kron( Vt_perp(mygrid(j,2),:), Ut(mygrid(j,1) ,:) );
            tildeA(j,:) = [Ab, Ad1, Ad2];
        end
        gamma = inv(tildeA' * tildeA) * tildeA' * y;
        %[gamma, flag] = lsqr(tildeA, y);
        hatB = reshape(gamma(1:r*r),[3,3]);
        hatD1 = reshape(gamma(r^2+1: r^2+r*(p1-r) ), [p1-r, r] );
        hatD2 = reshape(gamma( r^2+r*(p1-r)+1 : (p1+p2-r)*r ), [r, p2-r]);
        hatD2 = hatD2';
        % Gauss-Newton step on the manifold
        hatXu = Ut * hatB + Ut_perp* hatD1;
        hatXv = Vt * hatB' + Vt_perp * hatD2;
        % retraction step (orthographic retraction)
        hatX = hatXu * inv(hatB) * hatXv';
        [Ut,~] = qr(hatXu,0);
        [Vt,~] = qr(hatXv,0);
        Ut_perp = null(Ut');
        Vt_perp = null(Vt');
        rela_err = norm(hatX - X, 'fro')/norm(X, 'fro');
        time = toc;
        iter_result = [i, rela_err, time];
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
