function w = two_step_pca(X, eta, max_iters)
    % Two-step PCA with explicit normalization - batch update
    
    [N, d] = size(X);
    
    % Initialize weight vec
    w = randn(d, 1);
    w = w / norm(w);
    
    for iter = 1:max_iters
        w_old = w;
        
        % Batch update: use all at once
        y = X * w;  % N x 1 output vector
        delta_w = eta * (X' * y) / N;
        w = w + delta_w;
        
        % Explicit norm
        w = w / norm(w);
        
        % Prnt every 500 iterations
        if mod(iter, 500) == 0
            change = norm(w - w_old);
            fprintf('  Iteration %d, change: %.2e\n', iter, change);
        end
    end
end