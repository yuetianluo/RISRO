% RISRO for Phase Retrieval
% Input: A: n-by-p matrix, the stack of the sensing vectors; y: observation; p: dimension; x: true signal; x0: initialization;
% iter-max: iteration max number,
% tol:tolerence to terminate
% succ_tol: tolerance for successful recovery  
% Here eigendecomposition is used as retraction.
function [error_matrix,succ_tag] = RISRO_Phase_Retrieval(A, y, p, x, x0, iter_max, tol,succ_tol)
    xt = x0;
    rela_err = norm(xt * xt' - x * x', 'fro')/norm(x * x', 'fro');
    ut = xt/norm(xt);
    ut_perp = null(ut');
    error_matrix = [0, rela_err,0];
    tic; 
    for i = 1:iter_max
        % A new way to compute the importance covariates
        At = A * ut; 
        A1 = At.^2;
        A2 = (A * ut_perp).* At;
        tildeA = [A1, 2*A2];
        gamma = (tildeA' * tildeA) \ tildeA' * y;
        %[gamma, flag] = lsqr(tildeA, y);
        v = ut_perp * gamma(2:p);
        middle_matrix = [gamma(1),norm(v);
                        norm(v), 0];
        [V,D] = eigs(middle_matrix);
        U = [ut, v/norm(v)] * V;
        ut = U(:,1);
        hatx_whole = D(1,1) * (ut) * (ut');
        ut_perp = null(ut');
        rela_err = norm(hatx_whole - x * x', 'fro')/norm(x * x', 'fro');
        time = toc;
        iter_result = [i, rela_err,time];
        error_matrix = vertcat(error_matrix, iter_result);
        if rela_err < tol || rela_err > 10
            break
        end
    end
    if rela_err < succ_tol
        succ_tag = 1;
    else
        succ_tag = 0;
    end
end