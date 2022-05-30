% RISRO/Riemannian Gauss-Newton for Rank Constrained Least squares
% And in this code, we output the use time and error after every iteration.
    % Input: A: tensor form, each slice is the covariates, y: response, p1, p2: dimension of the parameter matrix, r: rank, iter-max: iteration max number,
    % tol:tolerence to terminate
    % succ_tol: tolerance for successful recovery 
    % hatX:initialization, 
    % X: underlying parameter matrix
    % retra_type: the retraction type of the algorithm. Two choices are 'orthogra': orthographic retraction and 'svd': SVD projection retraction. 
function [error_matrix,succ_tag] = RISRO(A, y, r, p1, p2, hatX,X, iter_max, tol, succ_tol, retra_type)
    A1 = tenmat(A,1);
    [U0,~,V0] = svds(hatX,r);
    Ut = U0;
    Ut_perp = null(Ut');
    Vt = V0;
    Vt_perp = null(Vt');
    rela_err = norm(hatX - X, 'fro')/norm(X, 'fro'); 
    error_matrix = [0,rela_err,0];
    tic;
    for i = 1:iter_max
        myU = ttm(A,Ut',2);
        myV = ttm(A,Vt',3);
        tildeB = tenmat(ttm(myU,Vt',3),1);
        tildeD1 = tenmat(ttm(myV,Ut_perp',2),1);
        tildeD2 = tenmat(ttm(myU,Vt_perp',3),1);
        tildeA = [tildeB.data, tildeD1.data, tildeD2.data];
        gamma = (tildeA' * tildeA) \ (tildeA' * y);
        %[gamma, ~] = lsqr(tildeA, y,1e-15,200);
        hatB = reshape(gamma(1:r*r),[r,r]);
        hatD1 = reshape(gamma(r^2+1: r^2+r*(p1-r) ), [p1-r, r] );
        hatD2 = reshape(gamma( r^2+r*(p1-r)+1 : (p1+p2-r)*r ), [r, p2-r]);
        hatD2 = hatD2';
        if strcmp(retra_type,'orthogra')  
            % Gauss-Newton step on the manifold
            hatXu = Ut * hatB + Ut_perp* hatD1;
            hatXv = Vt * hatB' + Vt_perp * hatD2;
            % retraction step
            hatX = hatXu / hatB * hatXv';
            [Ut,~] = qr(hatXu,0);
            [Vt,~] = qr(hatXv,0);
        elseif strcmp(retra_type,'svd')
            middle_matrix = zeros(p1,p2);
            middle_matrix(1:r,1:r) = hatB;
            middle_matrix((r+1):p1,1:r) = hatD1;
            middle_matrix(1:r,(r+1):p2) = hatD2';
            [U_mid,sigma_mid,V_mid] = svds(middle_matrix,r);
            hatX = [Ut Ut_perp] * U_mid * sigma_mid * V_mid' * [Vt Vt_perp]';
            [Ut,~,Vt] = svds(hatX,r);
        end
        Ut_perp = null(Ut');
        Vt_perp = null(Vt');
        rela_err = norm(hatX - X, 'fro')/norm(X, 'fro'); 
        time = toc;
        iter_result = [i, rela_err,time];
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