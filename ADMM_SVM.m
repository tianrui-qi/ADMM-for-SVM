function [w,b,out] = ADMM_SVM(x, y, lam, opts)
    % Alternating Direction Method of Multipliers (ADMM) for solving SVM
   
    [p,N] = size(x); 
    if isfield(opts,'tol')      tol = opts.tol;           else tol = 1e-3;       end
    if isfield(opts,'maxit')    maxit = opts.maxit;       else maxit = 5000;     end
    if isfield(opts,'w0')       w0 = opts.w0;             else w0 = randn(p,1);  end
    if isfield(opts,'b0')       b0 = opts.b0;             else b0 = 0;           end
    if isfield(opts,'beta')     beta = opts.beta;         else beta = 1;         end
    
    % constant
    X = transpose(x);
    X(:, p+1) = 1;
    X = y .* X;
    Q = eye(p);
    Q(p+1, p+1) = 0;
    left = (lam / beta) * Q + transpose(X) * X;
    
    % parameter
    W = [ w0 ; b0 ];    % dependent variable for subproblem 1
    T = 1 - X * W;      % dependent variable for subproblem 2
    u = zeros(N,1);     % Lagrangian multiplier
    
    % historical residual
    hist_pres = [];     % save historical primal residual
    hist_dres = [];     % save historical dual   residual
    
    iter = 0;
    while true
        iter = iter + 1;
        if iter >= maxit
            break
        end
        
        old_W = W;
        old_T = T;
        
        % update W
        W = left \ ( - transpose(X) * ( (1/beta) * u + T - 1 ) );
                
        % update T
        for i = 1:N
            C_i =  - (u(i) / beta) - ( X(i, :) * W ) + 1;
            if (1 / beta) < C_i
                T(i) = C_i - (1 / beta);
            elseif (0 <= C_i) && (C_i <= (1 / beta))
                T(i) = 0;
            else
                T(i) = C_i;
            end
        end
       
        % compute primal residual and save to hist_pres
        pres = norm(T + (X * W) - 1);
        hist_pres = [hist_pres; pres];
        
        % compute the dual residual and save to hist_dres
        dres = beta * norm( transpose(X) * ( T - old_T ) );
        % dres = beta * norm( X * ( W - old_W ) );
        hist_dres = [hist_dres; dres];
        
        % fprintf('out iter = %d, pres = %5.4e, dres = %5.4e\n', iter, pres, dres);

        if max(pres, dres) <= tol
            break
        end
        
        % update Lagrangian multiplier
        u = u + beta * (T + X * W - 1);
    end
    
    w = W(1:p);
    b = W(end);
    out.hist_pres = hist_pres;
    out.hist_dres = hist_dres;

end