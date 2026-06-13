function [w, errors] = iterative_pca_one_step(X, eta, max_epochs, tolerance)
    % One-step learning using Oja's rule
    
    [N, d] = size(X);
    
    % Initialize weight vec
    w = randn(d, 1);
    w = w / norm(w);
    
    errors = zeros(max_epochs, 1);
    
    for epoch = 1:max_epochs
        w_old = w;
        
        % Update (online learning)
        for i = 1:N
            x = X(i, :)';
            y = x' * w;  % output
            % Oja's rule: w_new = w + eta * y * (x - y * w)
            % This is the stabilized version that prevents blowing up
            w = w + eta * y * (x - y * w);
        end
        
        % Normalize to unit length
        w = w / norm(w);
        
        % Compute change for convergence monitoring
        change = norm(w - w_old);
        errors(epoch) = change;
        
        % Check Convgc
        if change < tolerance
            fprintf('  One-step converged after %d epochs (change: %.2e)\n', epoch, change);
            errors = errors(1:epoch);
            break;
        end
        
        % Print progress every 20 epo
        if mod(epoch, 20) == 0
            fprintf('  One-step epoch %d, change: %.2e\n', epoch, change);
        end
    end
end