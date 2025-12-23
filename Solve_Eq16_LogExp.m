% Solve_Eq16_LogExp.m
%
% Solves Eq. (16) with kernel (17) directly for F_l(J):
%   [omega - l*Omega_1(J) - m*Omega_2(J)] F_l(J) =
%       G * F_{0,l}(J) * sum_{l'} int dJ' Pi_{l,l'}(J,J') F_{l'}(J')
%
% with Pi_{l,l'}(J,J') = int dalpha/(2*pi) N(alpha,m) W_l^*(J,alpha) W_{l'}(J',alpha).
% The operator is applied in factorized alpha-form to avoid building
% a huge dense Pi matrix.
%
% This script reuses the model/grid/orbit/basis construction from Matrix_LogExp.m
% but assembles the eigenproblem directly in the J-grid.
%
% Authors: E.V. Polyachenko, I.G. Shukhman (adapted)
% Date: December 2025

% clear; close all;

% -- Add paths if they exist (safe-guard) --
if exist('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/models','dir')
    addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/models');
end
if exist('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/Original_Matlab_C','dir')
    addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/Original_Matlab_C');
end

%% ==================== PARAMETERS ====================

% Azimuthal number
m = 2;

% Resonance terms: l = l_min : l_max
l_min = -5;
l_max = 5;
n_l = l_max - l_min + 1;
l_arr = l_min:l_max;

% Alpha grid (Fourier conjugate to log r)
N_alpha = 201;
alpha_max = 10;
alpha_arr = linspace(-alpha_max, alpha_max, N_alpha);
S_alpha = TrapezoidCoef(alpha_arr);           % quadrature weights for d alpha

% Rc grid (logarithmic, case 3)
NR = 51;
Rc_ = logspace(-2, log10(10), NR);
uRc_ = log(Rc_);
S_RC = Rc_ .* TrapezoidCoef(uRc_);            % weights for dRc (with measure)

% v grid (circularity)
Ne = 11;
Full_ON = 0;  % 1: include retrograde, 0: prograde only

% Model selection
model = 1;  % 1: Toomre-Zang n=4, m=2;  9: Isochrone JH

% Angle grid for orbit integration
nw = 10001;
nwa = 101;  % Reduced grid for W_l computation
nwai = (nw-1)/(nwa-1);
Ia = 1:nwai:nw;

G = 1;  % gravitational constant in code units

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
        sigma_r = sigma_u;
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
        v_min = 1e-3;
    end
    v = linspace(v_min, 1, Ne);
    S_e(iR, :) = TrapezoidCoef(v);
    v_iR(iR, :) = v;
end
v_iR(abs(v_iR) < 1e-14) = 1e-12;
SGNL = sign(v_iR);

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
w1 = nan(nwa, NR, Ne);
ra = nan(nwa, NR, Ne);
pha = nan(nwa, NR, Ne);

for iR = 1:NR
    R_i = Rc(iR, 1);
    kappa_iE = kappa(R_i);
    Omega_iE = Omega(R_i);

    for ie = 1:Ne
        E_i = E(iR, ie);
        L_j = L_m(iR, ie);
        L2_j = L2_m(iR, ie);

        % Circular orbits
        if abs(e(iR, ie)) < 1e-10
            Omega_1(iR, ie) = kappa_iE;
            Omega_2(iR, ie) = Omega_iE * SGNL(iR, ie);
            w1(:, iR, ie) = wa;
            ra(:, iR, ie) = R_i;
            pha(:, iR, ie) = wa * Omega_2(iR, ie) / Omega_1(iR, ie);
            continue;
        end

        % Epicyclic approximation
        if e(iR, ie) < 0.01
            a_j = (R2(iR, ie) - R1(iR, ie)) / 2;
            Omega_1(iR, ie) = kappa_iE;
            Omega_2(iR, ie) = Omega_iE * SGNL(iR, ie);
            w1(:, iR, ie) = wa;
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

        w1(:, iR, ie) = t(Ia) * Omega_1(iR, ie);
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

    if mod(iR, 10) == 0, fprintf('.'); end
end
fprintf(' done.\n');

%% ==================== JACOBIAN ====================

fprintf('Computing Jacobian for (Rc,v) -> (J,L)...\n');

Jac = zeros(NR, Ne);
for iR = 1:NR
    for ie = 1:Ne
        r1 = R1(iR, ie); r2 = R2(iR, ie); rc = Rc(iR, ie);
        E_val = E(iR, ie); L_val = L_m(iR, ie); omega1 = Omega_1(iR, ie);
        Jac(iR, ie) = 0;
        general_orbit = (abs(r2 - r1) > 1e-12) && (abs(L_val) > 1e-12);
        if general_orbit
            V1 = V(r1); V2 = V(r2); dV1 = dV(r1); dV2 = dV(r2);
            t1 = 2 * (E_val - V1) * r1 - dV1 * r1^2;
            t2 = 2 * (E_val - V2) * r2 - dV2 * r2^2;
            denominator = r2^2 - r1^2;
            det_jac = abs(t1 * t2 / denominator);
            Jac(iR, ie) = 2 * det_jac * rc / omega1 / abs(L_val);
        end
        radial_orbit = abs(L_val) <= 1e-12;
        if radial_orbit
            DelE = 2.0 * (E_val - V(r1));
            Jac(iR, ie) = sqrt(max(DelE, 0)) * dV(r2) * 2 * rc / omega1;
            if ~isfinite(Jac(iR, ie)), Jac(iR, ie) = 0; end
        end
    end
end

%% ==================== DISTRIBUTION FUNCTION ====================

fprintf('Computing DF and derivatives...\n');

F0 = zeros(NR, Ne); FE = zeros(NR, Ne); FL = zeros(NR, Ne);
for iR = 1:NR
    for ie = 1:Ne
        F0(iR, ie) = DF(E(iR, ie), L_m(iR, ie));
        FE(iR, ie) = DF_E(E(iR, ie), L_m(iR, ie));
        FL(iR, ie) = DF_L(E(iR, ie), L_m(iR, ie));
    end
end
F0(~isfinite(F0)) = 0; FE(~isfinite(FE)) = 0; FL(~isfinite(FL)) = 0;

%% ==================== COMPUTE W_l (Eq. 10) ====================

fprintf('Computing W_l basis functions (Eq. 10)...\n');

W_l = zeros(NR, Ne, N_alpha, n_l);  % complex
for iR = 1:NR
    for ie = 1:Ne
        w1_vals = squeeze(w1(:, iR, ie));
        r_vals = squeeze(ra(:, iR, ie));
        theta_vals = squeeze(pha(:, iR, ie));
        if any(~isfinite(r_vals)) || any(~isfinite(theta_vals)) || any(~isfinite(w1_vals))
            continue;
        end
        Sw1 = TrapezoidCoef(w1_vals')';
        Om2_Om1 = Omega_2(iR, ie) / Omega_1(iR, ie);
        phi_vals = Om2_Om1 * w1_vals - theta_vals;
        for i_l = 1:n_l
            l = l_arr(i_l);
            for i_alpha = 1:N_alpha
                alpha = alpha_arr(i_alpha);
                angle_part = l * w1_vals + m * phi_vals;
                log_r = log(r_vals);
                integrand = cos(angle_part) .* exp(-1i * alpha * log_r) ./ sqrt(r_vals);
                integrand(~isfinite(integrand)) = 0;
                W_l(iR, ie, i_alpha, i_l) = (1/pi) * sum(Sw1 .* integrand);
            end
        end
    end
    if mod(iR, 10) == 0, fprintf('.'); end
end
fprintf(' done.\n');

%% ==================== N(alpha, m) KERNEL ====================

fprintf('Computing N(alpha, m) kernel...\n');

ia = 1i * alpha_arr;
z1 = (m + 0.5 + ia) / 2; z2 = (m + 0.5 - ia) / 2;
z3 = (m + 1.5 + ia) / 2; z4 = (m + 1.5 - ia) / 2;
N_kernel = real(gamma_complex(z1) .* gamma_complex(z2) ./ ...
               (gamma_complex(z3) .* gamma_complex(z4))) * pi;

%% ==================== BUILD OPERATOR (E + B) ON J-GRID ====================

fprintf('Preparing operator (E + B) on J-grid via factorization...\n');

NJ = NR * Ne;
weights_J = (S_RC(:) * ones(1, Ne)) .* S_e .* Jac;  % NR-by-NE
weights_J = reshape(weights_J, [NJ, 1]);             % column vector

% Flatten Omega combinations and F_{0,l}
Omega1_vec = reshape(Omega_1, [NJ, 1]);
Omega2_vec = reshape(Omega_2, [NJ, 1]);

E_l_list = cell(n_l,1);        % each is NJx1
F0l_list = cell(n_l,1);        % each is NJx1
Wl_list = cell(n_l,1);         % each is NJxN_alpha (not conjugated)
Wl_conj_list = cell(n_l,1);    % each is NJxN_alpha (conjugated)

for i_l = 1:n_l
    l = l_arr(i_l);
    E_l_list{i_l} = l * Omega1_vec + m * Omega2_vec;
    % F_{0,l} = l*dF/dJ + m*dF/dL, with (E,L) vars:
    % dF/dJ|_L = Omega_1 * FE, and (dF/dL)|_J = Omega_2 * FE + FL
    % => F_{0,l} = (l*Omega_1 + m*Omega_2) * FE + m * FL
    F0l_list{i_l} = (l * Omega1_vec + m * Omega2_vec) .* reshape(FE, [NJ,1]) + m * reshape(FL, [NJ,1]);
    Wtmp = reshape(W_l(:,:,:,i_l), [NJ, N_alpha]);
    Wl_list{i_l} = Wtmp;
    Wl_conj_list{i_l} = conj(Wtmp);
end

weightAlpha = (S_alpha(:) .* N_kernel(:)) ;   % column vector N_alpha

%% ==================== EXPLICIT KERNEL Π AND FULL MATRIX A ====================

fprintf('Building explicit kernel Π_{l,lp}(J,Jp ) by integrating over alpha ...\n');

N_total = NJ * n_l;    % total size of the operator A

% Pre-allocate A (dense, real) — Π and A are real since N(α,m) is even
A = zeros(N_total, N_total);

% Precompute column weights for \int dJ' ...
WJ_row = weights_J(:)';  % 1 x NJ, multiplies columns of Π

% Block assembly
for i_l = 1:n_l
    % Row block indices for l
    row_s = (i_l-1)*NJ + 1; row_e = i_l*NJ;

    % Diagonal E contribution (only on blocks with l' = l)
    El = E_l_list{i_l};               % NJ x 1
    A(row_s:row_e, row_s:row_e) = A(row_s:row_e, row_s:row_e) + diag(El);

    % B contribution with all l'
    F0l = F0l_list{i_l};              % NJ x 1
    Wl_conj = Wl_conj_list{i_l};      % NJ x N_alpha

    for i_lp = 1:n_l
        col_s = (i_lp-1)*NJ + 1; col_e = i_lp*NJ;

        Wlp = Wl_list{i_lp};          % NJ x N_alpha
        % Π_{l,l'}(J,J') = ∑_α [N(α,m)/(2π) S_α] · Re{ W_l^*(J,α) W_{l'}(J',α) }
        % Build explicitly with an α-loop to make reality and weights obvious
        Pi_llp = zeros(NJ, NJ);       % real
        for ia = 1:N_alpha
            wa = weightAlpha(ia);     % (S_alpha*N_kernel)/(2π)
            u = Wl_conj(:, ia);       % W_l^*(J, α)
            v = Wlp(:, ia);           % W_{l'}(J', α)
            % Outer product contribution at this α (take real part explicitly)
            Pi_llp = Pi_llp + wa * real(u * (v.'));  % NJ x NJ
        end

        % Apply dJ' quadrature weights on columns: Π * diag(weights_J)
        % Pi_llp = Pi_llp .* (ones(NJ,1) * WJ_row);  % column-wise scaling
        Pi_llp = Pi_llp * diag(WJ_row);  % column-wise scaling

        % Left-multiply by diag(F0l): row-wise scaling
        % Block_B = G * ((F0l * ones(1, NJ)) .* Pi_llp);          % NJ x NJ
        Block_B = G * diag(F0l) * Pi_llp;

        % Add to A block (real)
A(row_s:row_e, col_s:col_e) = A(row_s:row_e, col_s:col_e) + Block_B;
    end
end

%% ==================== EIGEN-SOLVE (dense) ====================

fprintf('Solving eigenproblem: (E + B) F = omega F using eig ...\n');

[eigvecs, eigvals] = eig(A);
omega_arr = diag(eigvals);
plot(real(omega_arr)/m, imag(omega_arr), '.')
axis([0 1 0 .2])

% Sort by imaginary part (growth rate) descending
[~, idx_sort] = sort(imag(omega_arr), 'descend');
omega_arr = omega_arr(idx_sort);
fprintf('\n============ EIGENVALUES (Eq. 16) ============\n');
fprintf('Omega_p | gamma\n');
fprintf('----------------\n');
for i = 1:min(6, length(omega_arr))
    Omega_p = real(omega_arr(i)) / m;
    gamma = imag(omega_arr(i));
    fprintf('%12.6f           | %12.6f\n', Omega_p, gamma);
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
        S(1) = dx/3; S(end) = dx/3;
        for i = 2:2:n-1, S(i) = 4*dx/3; end
        for i = 3:2:n-2, S(i) = 2*dx/3; end
    else
        S = TrapezoidCoef(x);
    end
end
