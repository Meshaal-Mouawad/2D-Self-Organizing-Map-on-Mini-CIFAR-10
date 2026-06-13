function w = two_step_pca(X, eta, max_iters)
    % Two-step PCA with explicit normalization - batch update
    
    [N, d] = size(X);
    
    % Initialize weight vec
    w = randn(d, 1);
    w = w / norm(w);
    
    for iter = 1:max_iters
        w_old = w;
        
        % Batch update: samples at once
        y = X * w;  % N x 1 output vector
        delta_w = eta * (X' * y) / N;
        w = w + delta_w;
        
        % Expc norm
        w = w / norm(w);
        
        % Print progres/ 500
        if mod(iter, 500) == 0
            change = norm(w - w_old);
            fprintf('  Iteration %d, change: %.2e\n', iter, change);
        end
    end
end