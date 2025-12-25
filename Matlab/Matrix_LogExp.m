% Matrix_LogExp.m
%
% Linear Matrix Method for Normal Modes using Logarithmic-Exponential basis
%
% Based on the separable kernel approach (Eq. 28):
%   omega * D * C = (E + B) * C
%
% where D, E are block-diagonal in l, and B couples different l modes.
%
% Authors: E.V. Polyachenko, I.G. Shukhman
% Date: December 2025

% clear; close all;

addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/models');
addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/Original_Matlab_C');

%% ==================== PARAMETERS ====================

% Azimuthal number
m = 2;

% Resonance terms: l = l_min : l_max
l_min = -10;
l_max = 10;
n_l = l_max - l_min + 1;
l_arr = l_min:l_max;

% WNORM=1.59e5;
WNORM=1;

% Alpha grid (Fourier conjugate to log r), folded to alpha >= 0
N_alpha   = 31;
alpha_max = 10;
alpha_arr = linspace(-alpha_max, alpha_max, N_alpha);
S_alpha   = SimpsonCoef(alpha_arr);

% Rc grid (logarithmic, case 3)
NR = 201;
Rc_ = logspace(-3, 2, NR);
uRc_ = log(Rc_);
S_RC = Rc_ .* SimpsonCoef(uRc_);  % Simpson weights 

% v grid (circularity, uniform from v_min to 1)
Ne = 31;
Full_ON = 0;  % 1: include retrograde (v from -1), 0: prograde only (v from 0)

% Model selection
model = 1;  % 1: Toomre-Zang n=4, m=2;  9: Isochrone JH

% Angle grid for orbit integration
nw   = 10001;
nwa  = 101;  % Reduced grid for W_l computation
nwai = (nw-1)/(nwa-1);
Ia = 1:nwai:nw;

%%
NPh = NR*Ne;
%% ==================== MODEL SETUP ====================

fprintf('Setting up model...\n');

switch model
    case 1
        [V, dV, Omega, kappa, DF, DF_E, DF_L, Sigma_d, sigma_r] = GModel_toomre();
        titlestr = 'Toomre-Zang n=4, m=2';
        DF_type = 0;
        
    case 9
        [V, dV, Omega, kappa, DF, DF_E, DF_L, Sigma_d, Jr, sigma_u, Omega1E, Omega2EL] = GModel_Isochrone(12, 'JH');
        titlestr = 'Isochrone model JH m=2';
        DF_type = 0;
        sigma_r = sigma_u;  % Alias for consistency
end

fprintf('Model: %s\n', titlestr);

%% ==================== GRID SETUP ====================

fprintf('Setting up grids...\n');

% v grid (uniform, with Simpson weights)
S_e = zeros(NR, Ne);
v_iR = zeros(NR, Ne);

for iR = 1:NR
    if Full_ON
        v_min = -1;
    else
        v_min = 1e-4;
    end
    v = linspace(v_min, 1, Ne);
    S_e(iR, :) = SimpsonCoef(v);  % Use Simpson for better accuracy
    v_iR(iR, :) = v;
end

% Avoid exactly zero v (causes issues with L=0)
v_iR(abs(v_iR) < 1e-14) = 1e-12;
SGNL = sign(v_iR);

% Compute orbital elements
e = 1 - abs(v_iR);
Rc = repmat(Rc_', 1, Ne);
R1 = Rc .* (1 - e);
R2 = Rc .* (1 + e);

% Energy from orbital elements
E = (V(R2) .* R2.^2 - V(R1) .* R1.^2) ./ (R2.^2 - R1.^2);

% Handle circular orbits
IEn = find(~isfinite(E) & e < 1);
Rin = Rc(IEn);
E(IEn) = Rin .* dV(Rin) / 2 + V(Rin);

% Handle radial orbits
IEn = find(~isfinite(E));
Rin = R2(IEn);
E(IEn) = V(Rin);

% Angular momentum
L2_m = 2 * (E - V(R2)) .* R2.^2;
L2_m(L2_m < 1e-14) = 0;
L_m = sqrt(L2_m) .* SGNL;

%% ==================== ORBIT INTEGRATION ====================

fprintf('Computing orbits...\n');

eps_w = 1e-8;
w = linspace(eps_w, pi - eps_w, nw);
dw = w(2) - w(1);
cosw = cos(w);
sinw = sin(w);
wa = w(Ia);

Omega_1 = zeros(NR, Ne);
Omega_2 = zeros(NR, Ne);
w1 = nan(nwa, NR, Ne);   % Actual angle variable w_1 (non-uniform in general)
ra = nan(nwa, NR, Ne);
pha = nan(nwa, NR, Ne);

for iR = 1:NR
    R_i = Rc(iR, 1);
    kappa_iE = kappa(R_i);
    Omega_iE = Omega(R_i);
    
    for ie = 1:Ne
        E_i  = E(iR, ie);
        L_j  = L_m(iR, ie);
        L2_j = L2_m(iR, ie);
        
        % Circular orbits
        if abs(e(iR, ie)) < 1e-10
            Omega_1(iR, ie) = kappa_iE;
            Omega_2(iR, ie) = Omega_iE * SGNL(iR, ie);
            w1(:, iR, ie) = wa;  % w1 = wa for circular
            ra(:, iR, ie) = R_i;
            pha(:, iR, ie) = wa * Omega_2(iR, ie) / Omega_1(iR, ie);
            continue;
        end
        
        % Epicyclic approximation
        if e(iR, ie) < 0.01
            a_j = (R2(iR, ie) - R1(iR, ie)) / 2;
            Omega_1(iR, ie) = kappa_iE;
            Omega_2(iR, ie) = Omega_iE * SGNL(iR, ie);
            w1(:, iR, ie) = wa;  % w1 ~ wa for epicyclic
            ra(:, iR, ie) = R_i - a_j * cosw(Ia);
            pha(:, iR, ie) = wa * Omega_2(iR, ie) / Omega_1(iR, ie) + ...
                2 * Omega_iE / kappa_iE * a_j / R_i * sinw(Ia) * SGNL(iR, ie);
            continue;
        end
        
        % General orbit integration
        r1 = R1(iR, ie);
        r2 = R2(iR, ie);
        rs = (r1 + r2) / 2;
        drs = (r2 - r1) / 2;
        xs = rs - drs * cosw;
        
        if L2_j > 1e-12
            rvr = sqrt(2 * (E_i - V(xs)) .* xs.^2 - L2_j);
            svr = sinw .* xs ./ rvr;
            svr(1) = svr(2) * 2 - svr(3);
            svr(end) = svr(end-1) * 2 - svr(end-2);
        else
            vr = sqrt(2 * (E_i - V(xs)));
            svr = sinw ./ vr;
            svr(end) = svr(end-1) * 2 - svr(end-2);
        end
        
        dt1 = drs * dw .* svr;
        dt2 = zeros(1, nw);
        dt2(2:end) = (dt1(1:end-1) + dt1(2:end)) / 2;
        t = cumsum(dt2);
        Omega_1(iR, ie) = pi / t(end);
        
        w1(:, iR, ie) = t(Ia) * Omega_1(iR, ie);  % w1 = Omega_1 * t (non-uniform!)
        ra(:, iR, ie) = xs(Ia);
        
        % Azimuthal angle integration
        if abs(1 - e(iR, ie)) > 1e-10 && L2_j > 1e-12
            svr = sinw ./ rvr;
            svr(1) = svr(2) * 2 - svr(3);
            svr(end) = svr(end-1) * 2 - svr(end-2);
            dt3 = drs * dw .* svr ./ xs;
            dt4 = zeros(1, nw);
            dt4(2:end) = (dt3(1:end-1) + dt3(2:end)) / 2;
            phi = cumsum(dt4);
            ph = L_j * phi(Ia);
        else
            ph = pi/2 * ones(size(Ia)) * SGNL(iR, ie);
            ph(1) = 0;
        end
        
        Omega_2(iR, ie) = Omega_1(iR, ie) * ph(end) / pi;
        pha(:, iR, ie) = ph;
    end
    
    if mod(iR, 10) == 0
        fprintf('.');
    end
end
fprintf(' done.\n');

%% ==================== JACOBIAN CALCULATION ====================

fprintf('Computing Jacobian for (Rc,v) -> (J,L) transformation...\n');

% The integration in Eqs. 21, 22, 24 is over dJ dL, not dRc dv
% We need the Jacobian: dJ dL = Jacobian * dRc dv

Jac = zeros(NR, Ne);

for iR = 1:NR
    for ie = 1:Ne
        r1 = R1(iR, ie);
        r2 = R2(iR, ie);
        rc = Rc(iR, ie);
        E_val = E(iR, ie);
        L_val = L_m(iR, ie);
        omega1 = Omega_1(iR, ie);
        
        Jac(iR, ie) = 0;
        
        % General orbit (not circular, not radial)
        general_orbit = (abs(r2 - r1) > 1e-12) && (abs(L_val) > 1e-12);
        if general_orbit
            V1 = V(r1);
            V2 = V(r2);
            dV1 = dV(r1);
            dV2 = dV(r2);
            
            % Partial derivatives
            t1 = 2 * (E_val - V1) * r1 - dV1 * r1^2;
            t2 = 2 * (E_val - V2) * r2 - dV2 * r2^2;
            denominator = r2^2 - r1^2;
            det_jac = abs(t1 * t2 / denominator);
            Jac(iR, ie) = 2 * det_jac * rc / omega1 / abs(L_val);
        end
        
        % Radial orbit
        radial_orbit = abs(L_val) <= 1e-12;
        if radial_orbit
            DelE = 2.0 * (E_val - V(r1));
            Jac(iR, ie) = sqrt(max(DelE, 0)) * dV(r2) * 2 * rc / omega1;
            if ~isfinite(Jac(iR, ie))
                Jac(iR, ie) = 0;
            end
        end

    end
end

%% ==================== DISTRIBUTION FUNCTION ====================

fprintf('Computing DF and derivatives...\n');

F0 = zeros(NR, Ne);
FE = zeros(NR, Ne);
FL = zeros(NR, Ne);

for iR = 1:NR
    for ie = 1:Ne
        F0(iR, ie) = DF(E(iR, ie), L_m(iR, ie));
        FE(iR, ie) = DF_E(E(iR, ie), L_m(iR, ie));
        FL(iR, ie) = DF_L(E(iR, ie), L_m(iR, ie));
    end
end

% Handle NaN/Inf values
F0(~isfinite(F0)) = 0;
FE(~isfinite(FE)) = 0;
FL(~isfinite(FL)) = 0;

%% ==================== COMPUTE W_l BASIS FUNCTIONS (Eq. 10) ====================

fprintf('Computing W_l basis functions (Eq. 10)...\n');

% From Eq. (5): phi(J, w) = (Omega_2/Omega_1) * w1 - theta
% where theta is the azimuthal angle stored in pha
% IMPORTANT: w1 is the actual angle variable, which is NON-UNIFORM in the
% parametrization. Must use non-uniform Trapezoid integration over w1.
%
% CORRECTED Eq. 10 with 1/pi normalization:
% W_l(J, alpha) = (1/pi) * integral_0^pi dw1 / sqrt(r) * cos[l*w1 + m*phi] * exp(-i*alpha*log(r))
%
% Note: w and phi are odd in w, but r is even, so W_l is COMPLEX.
%
% Index structure: W_l(iR, ie, i_alpha, i_l)

W_l = zeros(NR, Ne, N_alpha, n_l);  % Will be complex
U_l = zeros(NR, Ne, N_alpha, n_l);  % Will be complex

for iR = 1:NR
    for ie = 1:Ne
        w1_vals = squeeze(w1(:, iR, ie));      % Actual angle w1 (non-uniform!)
        r_vals = squeeze(ra(:, iR, ie));
        theta_vals = squeeze(pha(:, iR, ie));  % This is theta (azimuthal angle)
        
        if any(~isfinite(r_vals)) || any(~isfinite(theta_vals)) || any(~isfinite(w1_vals))
            continue;
        end
        
        % Integration weights for non-uniform w1 grid (Trapezoid)
        Sw1 = TrapezoidCoef(w1_vals')';  % Get weights for this specific w1 grid
        
        % Compute phi = Omega_2/Omega_1 * w1 - theta (Eq. 5)
        Om2_Om1 = Omega_2(iR, ie) / Omega_1(iR, ie);
        phi_vals = Om2_Om1 * w1_vals - theta_vals;  % phi uses w1, not wa!
        
        for i_l = 1:n_l
            l = l_arr(i_l);
            for i_alpha = 1:N_alpha
                alpha = alpha_arr(i_alpha);
                
                % cos[l*w1 + m*phi] * exp(-i*alpha*log(r)) / sqrt(r)
                angle_part = l * w1_vals + m * phi_vals;
                log_r = log(r_vals);

                integrand = cos(angle_part) .* exp(-1i * alpha * log_r) ./ sqrt(r_vals);
                integrand(~isfinite(integrand)) = 0;
                W_l(iR, ie, i_alpha, i_l) = (WNORM/pi) * sum(Sw1 .* integrand);

                integrand = cos(angle_part) .* exp(-1i * alpha * log_r) .* (r_vals);              integrand(~isfinite(integrand)) = 0;
                U_l(iR, ie, i_alpha, i_l) = (WNORM/pi) * sum(Sw1 .* integrand);
                
                % semilogx( w1_vals, real(integrand),  w1_vals, imag(integrand))
            end
        end
    end
    
    if mod(iR, 10) == 0
        fprintf('.');
    end
end
fprintf(' done.\n');

%% ==================== COMPUTE N(alpha, m) KERNEL ====================

fprintf('Computing N(alpha, m) kernel...\n');

ia = 1i * alpha_arr;

z1 = (m + 0.5 + ia) / 2;
z2 = (m + 0.5 - ia) / 2;
z3 = (m + 1.5 + ia) / 2;
z4 = (m + 1.5 - ia) / 2;

N_kernel = real(gamma_complex(z1) .* gamma_complex(z2) ./ ...
               (gamma_complex(z3) .* gamma_complex(z4))) * pi;

%% ==================== BUILD MATRICES D_l, E_l, F_l ====================

fprintf('Building D_l, E_l, F_l matrices...\n');

% F_{0,l} = l * dF0/dJ + m * dF0/dL
% In (E, L) coordinates: dF0/dJ = Omega_1 * dF0/dE
% Also, (dF0/dL)|_J = Omega_2 * FE + FL, so F_{0,l} = (l*Omega_1 + m*Omega_2) * FE + m * FL

D_l_mat = zeros(N_alpha, N_alpha, n_l);  % Will be complex
E_l_mat = zeros(N_alpha, N_alpha, n_l);  % Will be complex
F_l_mat = zeros(N_alpha, N_alpha, n_l);  % Will be complex

for i_l = 1:n_l
    l = l_arr(i_l);
    
    DJ     = repmat(S_RC.', 1, Ne) .* S_e .* Jac;
    DJ_vec  = reshape(DJ, NPh, 1);
    Om_vec = reshape(l * Omega_1 + m * Omega_2, NPh, 1);
    F0l_vec = reshape((l * Omega_1 + m * Omega_2) .* FE + m * FL, NPh, 1);

    W = reshape(W_l(:, :, :, i_l), NPh, N_alpha);
    fprintf('rank(W) = %d (vs N_alpha = %d), i_l=%d\n', rank(W), N_alpha, i_l);

    U = reshape(U_l(:, :, :, i_l), NPh, N_alpha);
    fprintf('rank(U) = %d (vs N_alpha = %d), i_l=%d\n', rank(U), N_alpha, i_l);

    D_l_mat(:, :, i_l) = W.' * diag(DJ_vec) * conj(W); 
    E_l_mat(:, :, i_l) = W.' * diag(DJ_vec.*Om_vec)  * conj(W); 
    F_l_mat(:, :, i_l) = W.' * diag(DJ_vec.*F0l_vec) * conj(W);

    fprintf('l = %d done\n', l);

    [Q, R, perm] = qr(W, 'vector');
    r = sum(abs(diag(R)) > tol);  % numerical rank

    dependent_cols = perm(r+1:end);  % indices of dependent α columns
    fprintf('Dependent α indices: ');
    disp(alpha_arr(dependent_cols));

end

%% ==================== BUILD B_{l,l'} MATRIX ====================

fprintf('Building B_{l,l''} matrix using factorization (Eq. 26)...\n');

G = 1;  % Gravitational constant 

B_mat = zeros(N_alpha * n_l, N_alpha * n_l);

for i_l = 1:n_l
    for i_lp = 1:n_l
        B_sum_mat = G* F_l_mat(:, :, i_l) * diag(N_kernel.*S_alpha) * D_l_mat(:, :, i_lp);
        Idx_row = (i_l  - 1) * N_alpha + (1:N_alpha);
        Idx_col = (i_lp - 1) * N_alpha + (1:N_alpha);
        B_mat(Idx_row, Idx_col) = B_sum_mat/WNORM^2;
    end
    fprintf('l = %d done\n', l_arr(i_l));
end

%% ==================== ASSEMBLE GLOBAL D AND E MATRICES ====================

fprintf('Assembling global matrices...\n');

% D and E are block-diagonal in l
N_total = N_alpha * n_l;
D_global = zeros(N_total, N_total);
E_global = zeros(N_total, N_total);

PR_D=1; 
PR_E=1; 
PR_F=1; 
for i_l = 1:n_l
    idx_start = (i_l - 1) * N_alpha + 1;
    idx_end = i_l * N_alpha;

    D_global(idx_start:idx_end, idx_start:idx_end) = D_l_mat(:, :, i_l);
    E_global(idx_start:idx_end, idx_start:idx_end) = E_l_mat(:, :, i_l);

    PR_D = PR_D * det(D_l_mat(:, :, i_l));
    PR_E = PR_E * det(E_l_mat(:, :, i_l));
    PR_F = PR_F * det(F_l_mat(:, :, i_l));
end
disp(num2str([abs(PR_D), abs(PR_E), abs(PR_F)], "|det(D)|=%.f, |det(E)|=%.f, |det(F)|=%.f"))
disp(num2str([abs(det(D_global)), abs(det(E_global))], "|det(D_global)|=%.f, |det(E_global)|=%.f"))

%%
log_det_D = 0;
log_det_E = 0;
log_det_F = 0;

for i_l = 1:n_l
    idx = (i_l - 1) * N_alpha + (1:N_alpha);
    D_global(idx, idx) = D_l_mat(:, :, i_l);
    E_global(idx, idx) = E_l_mat(:, :, i_l);

    [~, U_D] = lu(D_l_mat(:, :, i_l));
    [~, U_E] = lu(E_l_mat(:, :, i_l));
    [~, U_F] = lu(F_l_mat(:, :, i_l));
    
    log_det_D = log_det_D + sum(log(abs(diag(U_D))));
    log_det_E = log_det_E + sum(log(abs(diag(U_E))));
    log_det_F = log_det_F + sum(log(abs(diag(U_F))));
end

fprintf('log|det(D)|=%.2f, log|det(E)|=%.2f, log|det(F)|=%.2f\n', ...
        log_det_D, log_det_E, log_det_F);

%%
clf, hold on
for i_l=1:10:n_l,
    W = reshape(W_l(:, :, :, i_l), NPh, N_alpha);
    s = svd(W);
    plot(s / s(1), '.-');
    set(gca, 'YScale', 'log')
end
hold off
xlabel('Singular value index');
ylabel('Normalized singular value');
title('Singular value decay of W: Rc_ = logspace(-2, log10(10), NR); v_{min}=0.001');

%% ==================== SOLVE EIGENVALUE PROBLEM ====================

fprintf('Solving eigenvalue problem: omega * D * C = (E + B) * C\n');

% Generalized eigenvalue problem: (E + B) * C = omega * D * C
% Using MATLAB's eig: [V, Lambda] = eig(A, B) solves A*V = B*V*Lambda

RHS = E_global + B_mat;

try
    [eigvecs, eigvals] = eig(RHS, D_global);
    omega_arr = diag(eigvals);
    
    % Sort by imaginary part (growth rate)
    [~, idx_sort] = sort(imag(omega_arr), 'descend');
    omega_arr = omega_arr(idx_sort);
    eigvecs = eigvecs(:, idx_sort);
    
    fprintf('\n============ EIGENVALUES ============\n');
    fprintf('Omega_p (pattern speed) | gamma (growth rate)\n');
    fprintf('----------------------------------------\n');
    
    n_display = min(10, length(omega_arr));
    for i = 1:n_display
        Omega_p = real(omega_arr(i)) / m;
        gamma = imag(omega_arr(i));
        fprintf('%12.6f           | %12.6f\n', Omega_p, gamma);
    end

    plot(real(omega_arr)/m, imag(omega_arr), '.')
    % axis([0 2 0 .2])
    axis([0 1 0 .4])
    
catch ME
    fprintf('Eigenvalue computation failed: %s\n', ME.message);
    omega_arr = [];
    eigvecs = [];
end

%% ==================== HELPER FUNCTIONS ====================

function S = TrapezoidCoef(x)
    % Trapezoidal integration coefficients
    n = length(x);
    S = zeros(1, n);
    
    if n >= 2
        S(1) = (x(2) - x(1)) / 2;
        S(end) = (x(end) - x(end-1)) / 2;
        
        for i = 2:n-1
            S(i) = (x(i+1) - x(i-1)) / 2;
        end
    end
end

function S = SimpsonCoef(x)
    % Simpson's integration coefficients for uniform grids
    n = length(x);
    if n < 3
        S = TrapezoidCoef(x);
        return;
    end
    
    dx = x(2) - x(1);
    S = zeros(1, n);
    
    if mod(n, 2) == 1  % Odd number of points
        S(1) = dx/3;
        S(end) = dx/3;
        for i = 2:2:n-1
            S(i) = 4*dx/3;
        end
        for i = 3:2:n-2
            S(i) = 2*dx/3;
        end
    else  % Even number of points - use trapezoidal for last interval
        S = TrapezoidCoef(x);
    end
end
