% NL_NEWTON_SEARCH
%
% Newton-Raphson search for zeros of det(I - M·dα).
%
% Prerequisites: 
%   - Run NL_precompute.m first to create the 'data' struct
%   - Set initial guess: Omega_p_init, gamma_init (from NL_grid_scan or manual)
%
% Input (from workspace):
%   data         - struct from NL_precompute
%   Omega_p_init - initial pattern speed guess
%   gamma_init   - initial growth rate guess
%
% Output (to workspace):
%   Omega_p, gamma_val, history, and all intermediate variables

%% ==================== NEWTON PARAMETERS ====================
% Modify these as needed

max_iter = 30;
tol      = 1e-10;
delta    = 1e-6;   % Finite difference step

%% ==================== INITIAL GUESS ====================
% Use Omega_p_init, gamma_init from workspace (e.g., from NL_grid_scan)
% Or set manually:
% Omega_p_init = 0.44;
% gamma_init   = 0.13;

if ~exist('Omega_p_init', 'var') || ~exist('gamma_init', 'var')
    fprintf('WARNING: No initial guess found. Using Zang reference values.\n');
    Omega_p_init = 0.439;
    gamma_init   = 0.127;
end

Omega_p   = Omega_p_init;
gamma_val = gamma_init;

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

eyeN = eye(N_alpha);

%% ==================== NEWTON-RAPHSON ITERATION ====================

fprintf('\n=== Newton-Raphson search for det(I - M·dα) = 0 ===\n');
fprintf('Model: %s\n', data.titlestr);
fprintf('Initial: Ω_p = %.6f, γ = %.6f\n\n', Omega_p, gamma_val);

history = zeros(max_iter, 4);

for iter = 1:max_iter
    omega = m * Omega_p + 1i * gamma_val;
    
    % Build M matrix at current ω
    M = zeros(N_alpha, N_alpha);
    for i_l = 1:n_l
        denom = omega - Omega_res(:, i_l);
        small_idx = abs(denom) < 1e-12;
        denom(small_idx) = 1e-12 + 1e-12i;
        
        weight = DJ_vec .* F0l_all(:, i_l) ./ denom;
        W = W_l_mat(:, :, i_l);
        M = M + W.' * (weight .* conj(W));
    end
    M = G * M .* N_kernel;
    
    det_val = det(eyeN - M);
    
    % Residual
    F = [real(det_val); imag(det_val)];
    res_norm = norm(F);
    
    history(iter, :) = [Omega_p, gamma_val, real(det_val), imag(det_val)];
    
    fprintf('Iter %2d: Ω_p=%.10f  γ=%.10f  det=%.2e%+.2ei  |F|=%.2e\n', ...
            iter, Omega_p, gamma_val, real(det_val), imag(det_val), res_norm);
    
    if res_norm < tol
        fprintf('\n*** Converged! ***\n');
        history = history(1:iter, :);
        break;
    end
    
    % Jacobian via finite differences
    % ∂det/∂Ω_p
    omega_dOp = m * (Omega_p + delta) + 1i * gamma_val;
    M_dOp = zeros(N_alpha, N_alpha);
    for i_l = 1:n_l
        denom = omega_dOp - Omega_res(:, i_l);
        small_idx = abs(denom) < 1e-12;
        denom(small_idx) = 1e-12 + 1e-12i;
        weight = DJ_vec .* F0l_all(:, i_l) ./ denom;
        W = W_l_mat(:, :, i_l);
        M_dOp = M_dOp + W.' * (weight .* conj(W));
    end
    M_dOp = G * M_dOp .* N_kernel;
    det_dOp = det(eyeN - M_dOp);
    dF_dOp = ([real(det_dOp); imag(det_dOp)] - F) / delta;
    
    % ∂det/∂γ
    omega_dg = m * Omega_p + 1i * (gamma_val + delta);
    M_dg = zeros(N_alpha, N_alpha);
    for i_l = 1:n_l
        denom = omega_dg - Omega_res(:, i_l);
        small_idx = abs(denom) < 1e-12;
        denom(small_idx) = 1e-12 + 1e-12i;
        weight = DJ_vec .* F0l_all(:, i_l) ./ denom;
        W = W_l_mat(:, :, i_l);
        M_dg = M_dg + W.' * (weight .* conj(W));
    end
    M_dg = G * M_dg .* N_kernel;
    det_dg = det(eyeN - M_dg);
    dF_dg = ([real(det_dg); imag(det_dg)] - F) / delta;
    
    J = [dF_dOp, dF_dg];
    
    % Check condition
    cond_J = cond(J);
    if cond_J > 1e10
        fprintf('         Warning: Jacobian ill-conditioned (cond = %.2e)\n', cond_J);
    end
    
    % Newton step with line search
    dx = J \ F;
    
    step = 1.0;
    for ls = 1:10
        Omega_p_new = Omega_p - step * dx(1);
        gamma_new = gamma_val - step * dx(2);
        
        if gamma_new > 1e-4 && Omega_p_new > 0 && Omega_p_new < 1
            omega_new = m * Omega_p_new + 1i * gamma_new;
            M_new = zeros(N_alpha, N_alpha);
            for i_l = 1:n_l
                denom = omega_new - Omega_res(:, i_l);
                small_idx = abs(denom) < 1e-12;
                denom(small_idx) = 1e-12 + 1e-12i;
                weight = DJ_vec .* F0l_all(:, i_l) ./ denom;
                W = W_l_mat(:, :, i_l);
                M_new = M_new + W.' * (weight .* conj(W));
            end
            M_new = G * M_new .* N_kernel;
            det_new = det(eyeN - M_new);
            F_new = [real(det_new); imag(det_new)];
            if norm(F_new) < res_norm || ls == 10
                break;
            end
        end
        step = step * 0.5;
    end
    
    if step < 1
        fprintf('         (line search: step = %.3f)\n', step);
    end
    
    Omega_p = Omega_p - step * dx(1);
    gamma_val = gamma_val - step * dx(2);
end

%% ==================== FINAL OUTPUT ====================

fprintf('\n=== Final Result ===\n');

% Format Omega_p and gamma with apostrophes every 3 digits after decimal point
Op_str = sprintf('%.10f', Omega_p);
gamma_str = sprintf('%.10f', gamma_val);
dot_pos_op = strfind(Op_str, '.');
Op_int = Op_str(1:dot_pos_op);
Op_dec = Op_str(dot_pos_op+1:end);
Op_formatted = [Op_int Op_dec(1:3) '''' Op_dec(4:6) '''' Op_dec(7:10)];
dot_pos_g = strfind(gamma_str, '.');
gamma_int = gamma_str(1:dot_pos_g);
gamma_dec = gamma_str(dot_pos_g+1:end);
gamma_formatted = [gamma_int gamma_dec(1:3) '''' gamma_dec(4:6) '''' gamma_dec(7:10)];

% Print parameter table with Omega_p and gamma as last columns
fprintf('|---------|---------|-------|-------|---------|---------|----------|----------|---------|-----------|------------------|---------------------|\n');
fprintf('| N_alpha | α_max   | l_max | v_min | Rc_min  | Rc_max  | NW_orbit | NW       | N_R×Ne  | α-Rc-v    | Ω_p              | γ                   |\n');
fprintf('|---------|---------|-------|-------|---------|---------|----------|----------|---------|-----------|------------------|---------------------|\n');
fprintf('| %7d | %7.1f | %5d | %5.0e | %7.2f | %7.2f | %8d | %8d | %3d×%-3d | %-9s | %16s | %19s |\n', ...
    data.N_alpha, data.alpha_max, data.l_max, data.v_min, data.Rc_min, data.Rc_max, data.nw, data.nwa, data.NR, data.Ne, data.integ, Op_formatted, gamma_formatted);
fprintf('|---------|---------|-------|-------|---------|---------|----------|----------|---------|-----------|------------------|---------------------|\n');

%% ==================== EIGENVALUE CHECK ====================

% fprintf('\n=== Eigenvalue spectrum at final ω ===\n');
% 
% omega_final = m * Omega_p + 1i * gamma_val;
% M_final = zeros(N_alpha, N_alpha);
% for i_l = 1:n_l
%     denom = omega_final - Omega_res(:, i_l);
%     small_idx = abs(denom) < 1e-12;
%     denom(small_idx) = 1e-12 + 1e-12i;
%     weight = DJ_vec .* F0l_all(:, i_l) ./ denom;
%     W = W_l_mat(:, :, i_l);
%     M_final = M_final + W.' * (weight .* conj(W));
% end
% M_final = G * M_final .* N_kernel;
% 
% [V_eig, D_eig] = eig(M_final);
% eigvals = diag(D_eig);
% [~, idx_sort] = sort(abs(eigvals - 1));
% 
% fprintf('\nTop 6 eigenvalues closest to 1:\n');
% fprintf('  #  |   Re(λ)   |   Im(λ)   |   |λ-1|\n');
% fprintf('-----+-----------+-----------+---------\n');
% for k = 1:min(6, length(eigvals))
%     ev = eigvals(idx_sort(k));
%     fprintf('  %d  | %9.6f | %9.6f | %7.2e\n', k, real(ev), imag(ev), abs(ev-1));
% end

%% ==================== PLOT CONVERGENCE ====================

% figure('Name', 'Newton Convergence');
% 
% subplot(2,1,1);
% semilogy(1:size(history,1), sqrt(history(:,3).^2 + history(:,4).^2), ...
%          'b.-', 'LineWidth', 1.5, 'MarkerSize', 15);
% xlabel('Iteration');
% ylabel('|det(I - M)|');
% title('Newton Convergence');
% grid on;
% 
% subplot(2,1,2);
% plot(history(:,1), history(:,2), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
% hold on;
% plot(0.439426, 0.127181, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'k');
% plot(history(end,1), history(end,2), 'gs', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
% xlabel('\Omega_p');
% ylabel('\gamma');
% legend('Newton path', 'Zang reference', 'Final result', 'Location', 'best');
% title('Search trajectory in (\Omega_p, \gamma) space');
% grid on;
% hold off;
