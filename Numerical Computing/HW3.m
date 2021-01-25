function HW3
    matrix_solve(10)
    matrix_solve(50)
    matrix_solve(100)
    matrix_solve(500)
    matrix_solve(1000)  
    matrix_solve(2000)
end

function matrix_solve(n)
    A = eye(n);
    for i = 1:n
        A(1, i) = i;
        A(i, 1) = i;
    end
    x = ones(n);
    b = A*x;
    x_c = A\b;
    r = A*x_c;
    rel_error = norm(x - x_c, Inf) / norm(x, Inf);
    K = cond(A, Inf);
    rel_residual = norm(r, Inf) / norm(b, Inf);
    heuristic = eps * K;
    fprintf('\n n = %d     ||x - x_c||/||x|| = %.3g     K(A) = %.3g     ||r||/||b|| = %.3g     eK(A) = %.3g',n,rel_error,K,rel_residual, heuristic) 
end
   
