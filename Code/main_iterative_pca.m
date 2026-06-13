%% ECE8493 Project 4: Iterative PCA on MNIST
% We implement iterative PCA using one-step and two-step learning
% We focus on a single digit (digit 0) to reduce workload

clear; close all; clc;
rng(42);

%% Load 
fprintf('Loading MNIST data...\n');
load('MNIST_train.mat');  % gives imagestrain (28x28x60000), labelstrain (60000x1)
load('MNIST_test.mat');   % gives images (28x28x10000), labels (10000x1)

%% Reshape images
fprintf('Reshaping images...\n');

% Training data: 28x28x60000 -> 60000
X_train = reshape(imagestrain, 28*28, size(imagestrain, 3))';
Y_train = labelstrain;

% Test data: 28x28x10000 -> 10000
X_test = reshape(images, 28*28, size(images, 3))';
Y_test = labels;

fprintf('Training data size: %d x %d\n', size(X_train, 1), size(X_train, 2));
fprintf('Test data size: %d x %d\n', size(X_test, 1), size(X_test, 2));

%% Digit 0 to reduce workload
fprintf('\nSelecting only digit 0 images...\n');

% Find indices of digit 0 
idx_train_0 = (Y_train == 0);
X_train_0 = X_train(idx_train_0, :);
fprintf('Training: %d images of digit 0\n', size(X_train_0, 1));

% Find indices of digit 0 in test set
idx_test_0 = (Y_test == 0);
X_test_0 = X_test(idx_test_0, :);
fprintf('Testing: %d images of digit 0\n', size(X_test_0, 1));

% Center the data
fprintf('Centering the data...\n');
mean_face = mean(X_train_0, 1);
X_centered = X_train_0 - mean_face;

%% params for iterative PCA
eta = 0.001;        % Small learning rate
max_iters = 5000;   % Number of iterations

fprintf('\n========== PART 1: FIRST PRINCIPAL COMPONENT ==========\n');

% Method 1
fprintf('\n--- Method 1: One-step learning (Oja) ---\n');
w1_one_step = oja_pca(X_centered, eta, max_iters);
fprintf('First PC found using one-step learning\n');

% Method 2
fprintf('\n--- Method 2: Two-step learning with normalization ---\n');
w1_two_step = two_step_pca(X_centered, eta, max_iters);
fprintf('First PC found using two-step learning\n');

% Compare two methods
fprintf('\n--- Comparing the two methods ---\n');
dot_product = abs(dot(w1_one_step, w1_two_step));
fprintf('Dot product between the two PCs: %.6f\n', dot_product);

% Reconstruct the data using the 1st PC
fprintf('\n--- Reconstruction using first PC ---\n');
recon_one_step = X_centered * w1_one_step * w1_one_step' + mean_face;
error_one_step = mean(mean((X_train_0 - recon_one_step).^2));
fprintf('Reconstruction error (one-step): %.6f\n', error_one_step);

%% ========== PART 2: SECOND PRINCIPAL COMPONENT ==========
fprintf('\n========== PART 2: SECOND PRINCIPAL COMPONENT ==========\n');

% Remove the contribution of the first PC
X_removed = X_centered - (X_centered * w1_one_step) * w1_one_step';

% Find second PC using one-step
fprintf('\n--- Method 1: One-step learning for second PC ---\n');
w2_one_step = oja_pca(X_removed, eta, max_iters);
fprintf('Second PC found using one-step learning\n');

% orthogonality with first PC
ortho = abs(dot(w1_one_step, w2_one_step));
fprintf('Orthogonality (|dot|): %.6f\n', ortho);

fprintf('\n--- Reconstruction using first two PCs ---\n');
recon_2pc = X_centered * (w1_one_step * w1_one_step' + w2_one_step * w2_one_step') + mean_face;
error_2pc = mean(mean((X_train_0 - recon_2pc).^2));
fprintf('Reconstruction error with 2 PCs: %.6f\n', error_2pc);

% Compare with 1 PC err
improvement = (error_one_step - error_2pc) / error_one_step * 100;
fprintf('Error reduction: %.2f%%\n', improvement);

fprintf('\n========== VISUALIZATION ==========\n');

pc1_image = reshape(w1_one_step, [28, 28]);
pc2_image = reshape(w2_one_step, [28, 28]);

figure(1);
for i = 1:min(9, size(X_train_0, 1))
    subplot(3, 3, i);
    img = reshape(X_train_0(i, :), [28, 28]);
    imagesc(img);
    colormap(gray);
    axis off;
    title(sprintf('Original %d', i));
end
sgtitle('Sample Original Digit 0 Images');
saveas(gcf, 'pca_original_samples.png');

figure(2);
subplot(1, 2, 1);
pc1_display = (pc1_image - min(pc1_image(:))) / (max(pc1_image(:)) - min(pc1_image(:)));
imagesc(pc1_display);
colormap(gray);
axis off;
title('First Principal Component');
colorbar;

subplot(1, 2, 2);
pc2_display = (pc2_image - min(pc2_image(:))) / (max(pc2_image(:)) - min(pc2_image(:)));
imagesc(pc2_display);
colormap(gray);
axis off;
title('Second Principal Component');
colorbar;
saveas(gcf, 'pca_pcs.png');

if size(X_train_0, 1) >= 4
    figure(3);
    for i = 1:4
        % Original
        subplot(2, 4, i);
        img_orig = reshape(X_train_0(i, :), [28, 28]);
        imagesc(img_orig);
        colormap(gray);
        axis off;
        title(sprintf('Original %d', i));
        
        % Reconstruction using 2 PCs
        subplot(2, 4, i+4);
        img_recon = reshape(recon_2pc(i, :), [28, 28]);
        imagesc(img_recon);
        colormap(gray);
        axis off;
        title(sprintf('Recon %d', i));
    end
    sgtitle('Original vs Reconstruction (2 PCs, one-step method)');
    saveas(gcf, 'pca_reconstruction.png');
end

fprintf('\n=== DONE ===\n');
fprintf('Note: If errors are NaN, the algorithm may need more iterations.\n');