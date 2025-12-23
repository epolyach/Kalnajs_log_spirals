% NL_GRID_SCAN
%
% Grid scan for zeros of det(I - M·dα) over complex ω plane.
%
% Prerequisites: Run NL_precompute.m first to create the 'data' struct
%ri
% Input (from workspace):
%   data - struct from NL_precompute
%
% Output (to workspace):
%   det_grid, Omega_p_grid, gamma_grid, Omega_p_init, gamma_init

%% ==================== SCAN PARAMETERS ====================
% Modify these as needed

n_Op        = 41;     % Ω_p grid points
n_gamma     = 20;     % γ grid points  
Omega_p_min = 0.2;    % Min pattern speed
Omega_p_max = 0.6;    % Max pattern speed
gamma_min   = 0.01;   % Min growth rate
gamma_max   = 0.2;    % Max growth rate

%% ==================== EXTRACT DATA ====================

m         = data.m;
G         = data.G;
N_alpha   = data.N_alpha;
n_l       = data.n_l;
NPh       = data.NPh;
N_kernel  = data.N_kernel;
DJ_vec    = data.DJ_vec;
F0l_all   = data.F0l_all;
Omega_res = data.Omega_res;
W_l_mat   = data.W_l_mat;

%% ==================== GRID SETUP ====================

Omega_p_grid = linspace(Omega_p_min, Omega_p_max, n_Op);
gamma_grid   = linspace(gamma_min, gamma_max, n_gamma);

det_grid = zeros(n_gamma, n_Op);

fprintf('\n=== Grid scan for det(I - M·dα) zeros ===\n');
fprintf('Model: %s\n', data.titlestr);
fprintf('Scanning %d × %d = %d ω values...\n', n_Op, n_gamma, n_Op*n_gamma);
tic;

eyeN = eye(N_alpha);

%% ==================== SCAN LOOP ====================

for i_Op = 1:n_Op
    for i_g = 1:n_gamma
        omega = m * Omega_p_grid(i_Op) + 1i * gamma_grid(i_g);
        
        % Build M matrix (Eq. 45)
        M = zeros(N_alpha, N_alpha);
        for i_l = 1:n_l
            denom = omega - Omega_res(:, i_l);           
            weight = DJ_vec .* F0l_all(:, i_l) ./ denom;
            W = W_l_mat(:, :, i_l);
            M = M + W.' * (weight .* conj(W));
        end
        M = G * M .* N_kernel;
        
        det_grid(i_g, i_Op) = det(eyeN - M);
    end
    
    if mod(i_Op, 5) == 0
        fprintf('.');
    end
end
fprintf(' done (%.1f s)\n', toc);

%% ==================== FIND MINIMUM ====================

[min_val, min_idx] = min(abs(det_grid(:)));
[i_g_min, i_Op_min] = ind2sub(size(det_grid), min_idx);

Omega_p_init = Omega_p_grid(i_Op_min);
gamma_init   = gamma_grid(i_g_min);

fprintf('\nMinimum |det| = %.2e at Ω_p = %.4f, γ = %.4f\n', ...
        min_val, Omega_p_init, gamma_init);

%% ==================== PLOT RESULTS ====================

figure('Name', 'det(I - M·dα) grid scan');

contourf(Omega_p_grid, gamma_grid, log10(abs(det_grid)), 30);
xlabel('\Omega_p = Re(\omega)/m');
ylabel('\gamma = Im(\omega)');
title('log_{10}|det(I - M \cdot d\alpha)|');
colorbar;
hold on;
plot(Omega_p_init, gamma_init, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'w', 'LineWidth', 2);
plot(0.439, 0.127, 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
text(0.439+0.02, 0.127, 'Zang', 'Color', 'r');
hold off;


fprintf('\nInitial guess for Newton: Omega_p_init = %.4f, gamma_init = %.4f\n', ...
        Omega_p_init, gamma_init);
