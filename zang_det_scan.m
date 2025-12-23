% ZANG_DET_SCAN
% ---------------------------------------------------------------
% Compute det(M(ω) - I) on a grid of complex frequencies ω
% and plot |det| as a surface to locate eigenvalues (zeros).
%
% Grid:
%   Re(ω)/m = Ω_p ∈ [0.3, 0.5]  (51 nodes)
%   Im(ω) = s ∈ [0.01, 1] logarithmic (41 nodes)
%
% The mode eigenvalue condition is det(M - I) = 0.
% ---------------------------------------------------------------

% --------------- Global model parameters -----------------------
m      = 2;          % angular harmonic
Ncut   = 4;          % cut-out index N
q      = 6;

Omega_p = 0.4394;
gamma_t   = 0.1270;
% Omega_p = 0.141;
% gamma_t   = 0.066;

sigmaU = 1/sqrt(q+1);   % radial velocity dispersion σ_u / V0  (Q≈1)
a_par  = q;  % 'a' in Zang's DF: a = (V0/σ_u)^2 - 1

% Wavenumber grid α (log-spiral basis)
nAlpha   = 201;       % reduce for faster scanning
alphaMax = 10.0;
alpha    = linspace(-alphaMax, alphaMax, nAlpha);
dAlpha   = alpha(2)-alpha(1);

% Eccentric-velocity grid η = U/V0
nEta   = 80;
etaMax = 4.0*sigmaU;
eta    = linspace(1e-6, etaMax, nEta);
wEta   = eta_spacing_weights(eta);

% Radial Fourier harmonics ℓ range
ellMin = -10;
ellMax =  10;
ellVec = ellMin:ellMax;
nEll   = numel(ellVec);

% Number of orbital phase samples per radial period
nPsi   = 128;

% Build Kalnajs gravity factor K(α,m)
Kalpha  = kalnajs_K(alpha, m);

% ---------- Precompute orbits and Fourier coefficients ----------
fprintf('Precomputing orbits and Q_{ℓm}(α;η)...\n');
orbitData = cell(1,nEta);
for k = 1:nEta
    orbitData{k} = compute_orbit_data(eta(k), nPsi);
end

Q_lmk = compute_Q_lm(alpha, eta, orbitData, m, ellVec);

% ---------- Compute C_a tilde normalization (analytic, eq. 2.63) ----------
Ca_tilde_inv = sqrt(pi) * ((a_par+1)/2)^(-a_par/2) * gamma((a_par+1)/2) * exp((a_par+1)/2) / (a_par+1);
Ca_tilde     = 1.0 / Ca_tilde_inv / 2;

% ==========================================================================
% PRECOMPUTE ALL ω-INDEPENDENT QUANTITIES
% ==========================================================================
fprintf('Precomputing ω-independent quantities...\n');
tic;

% --- h-grid for F_lm integration ---
hmax = 12.0;
Nh   = 8192/2;
h    = linspace(-hmax, hmax, Nh);
dh_val = h(2) - h(1);

Hhat_h       = 1 ./ (1 + exp(-Ncut * h));  % 1 x Nh
Hhat_prime_h = Ncut * exp(-Ncut * h) ./ (1 + exp(-Ncut * h)).^2;  % 1 x Nh
exp_h        = exp(h);  % 1 x Nh

% --- ν values and exp_matrix for Fourier transform ---
nuVals = (-nAlpha+1:nAlpha-1) * dAlpha;  % 2*nAlpha-1 values
nNu    = numel(nuVals);
nuVals_col = nuVals(:);
exp_matrix = exp(-1i * nuVals_col * h);  % nNu x Nh

% --- Index matrix for F_lm → matrix mapping ---
[JJ, II] = meshgrid(1:nAlpha, 1:nAlpha);
idxMat_flat = (II(:) - JJ(:)) + nAlpha;  % nAlpha^2 x 1

% --- Extract orbital frequencies ---
kappa_vec = zeros(1, nEta);
Omega_vec = zeros(1, nEta);
I0_vec    = zeros(1, nEta);
for kEta = 1:nEta
    orb = orbitData{kEta};
    kappa_vec(kEta) = orb.kappa;
    Omega_vec(kEta) = orb.Omega;
    I0_vec(kEta)    = orb.I0;
end

% --- Weights for η integration ---
gauss_eta_vec = exp(-eta.^2 / (2*sigmaU^2));
weight_vec    = Ca_tilde * I0_vec .* wEta .* eta .* gauss_eta_vec;  % 1 x nEta

% --- Precompute QQ̄ outer products for all (ℓ, η) ---
% QQbar_flat(:,iEll,kEta) is the flattened nAlpha^2 version
QQbar_flat = zeros(nAlpha*nAlpha, nEll, nEta);
for kEta = 1:nEta
    for iEll = 1:nEll
        Q_vec = Q_lmk(iEll, :, kEta);  % 1 x nAlpha
        QQbar = (Q_vec.' * conj(Q_vec)).';
        QQbar_flat(:,iEll,kEta) = QQbar(:);
    end
end

% --- Precompute F_lm NUMERATOR for all (ℓ, η) ---
% numerator(ℓ,η,h) = [(a+1)(freq_term) - am] * Ĥ(h) - m * Ĥ'(h)
% This is ω-INDEPENDENT!
% Store transposed for efficient batched access: (Nh, nEll, nEta)
freq_term_all = ellVec(:) * kappa_vec + m * Omega_vec;  % nEll x nEta
numerator_coeff = (a_par+1) * freq_term_all - a_par * m;  % nEll x nEta
% numerator_all(h, ℓ, η) for efficient column-major access
numerator_all = zeros(Nh, nEll, nEta);
for kEta = 1:nEta
    numerator_all(:,:,kEta) = Hhat_h(:) .* numerator_coeff(:,kEta).' - m * Hhat_prime_h(:);
end

fprintf('Precomputation done in %.1f s\n', toc);

% ---------- Define ω grid ----------
nOmegaP = 21;   % Real axis: Ω_p = Re(ω)/m
nS      = 21;   % Imag axis: s = Im(ω) (log scale)

Omega_p_grid = linspace(0.0, 0.5, nOmegaP);
log10_s_grid = linspace(-2, 0, nS);
s_grid       = 10.^log10_s_grid;

% Storage for determinant values
detM = zeros(nS, nOmegaP);

% ---------- Scan over ω grid ----------
% Flatten omega grid for parfor compatibility
omega_grid = zeros(nOmegaP, nS);
for iOp = 1:nOmegaP
    for iS = 1:nS
        omega_grid(iOp, iS) = m * Omega_p_grid(iOp) + 1i * s_grid(iS);
    end
end
omega_flat = omega_grid(:);  % nOmegaP*nS x 1
nOmega = numel(omega_flat);

fprintf('\nScanning %d × %d = %d omega values...\n', nOmegaP, nS, nOmega);
tic;

% Preallocate identity matrix
eyeN = eye(nAlpha);

% Use parfor for parallelization (falls back to for if no PCT)
detM_flat = zeros(nOmega, 1);
parfor iOmega = 1:nOmega
    omega = omega_flat(iOmega);
    
    % Build kernel using precomputed data
    M = build_kernel_fast(omega, nAlpha, nEta, nEll, nNu, Nh, ...
                          freq_term_all, numerator_all, exp_h, exp_matrix, dh_val, ...
                          idxMat_flat, QQbar_flat, weight_vec, Kalpha, dAlpha);
    
    % Compute det(M - I)
    detM_flat(iOmega) = det(M - eyeN);
end

% Reshape back to grid
detM = reshape(detM_flat, nOmegaP, nS).';

fprintf('Total time: %.1f s\n', toc);

%% ---------- Plot results ----------
figure('Name', 'det(M - I) scan');

contourf(Omega_p_grid, log10_s_grid, log10(abs(detM)), 30);
xlabel('\Omega_p = Re(\omega)/m');
ylabel('log_{10}(s)');
title('log_{10}|det(M - I)| contours');
colorbar;
hold on;

% Mark Zang's reference point
plot(Omega_p, log10(gamma_t), 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
text(Omega_p, log10(gamma_t)+0.1, 'Zang', 'Color', 'r', 'FontWeight', 'bold');
hold off;
% saveas(gcf, 'tz_q2_N4.png', 'png')

% Save data for later analysis
save('det_scan_results.mat', 'Omega_p_grid', 's_grid', 'log10_s_grid', 'detM', 'm');
fprintf('Results saved to det_scan_results.mat\n');


%% =================================================================
% Helper: simple trapezoidal weights for η‑grid (vectorized)
% =================================================================
function w = eta_spacing_weights(eta)
n = numel(eta);
if n == 1
    w = 1.0;
    return;
end
d = diff(eta);
w = zeros(size(eta));
w(1)   = 0.5*d(1);
w(end) = 0.5*d(end);
w(2:end-1) = 0.5*(d(1:end-1) + d(2:end));
end


% =================================================================
% Kalnajs gravity factor K(α,m)  [eq. (3.10)]
% =================================================================
function K = kalnajs_K(alpha, m)
alpha = alpha(:).';
ia    = 1i*alpha;

z1 = (m + 0.5 + ia)/2;
z2 = (m + 0.5 - ia)/2;
z3 = (m + 1.5 + ia)/2;
z4 = (m + 1.5 - ia)/2;

K = cgamma(z1).*cgamma(z2) ./ (cgamma(z3).*cgamma(z4));
K = real(K);
end


% =================================================================
% Complex gamma function using Lanczos approximation (VECTORIZED)
% =================================================================
function y = cgamma(z)
g = 7;
p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, ...
     771.32342877765313, -176.61502916214059, 12.507343278686905, ...
     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

original_shape = size(z);
z = z(:);

need_reflect = real(z) < 0.5;
z_work = z;
z_work(need_reflect) = 1 - z(need_reflect);

zm1 = z_work - 1;
denoms = zm1 + (1:8);
x = p(1) + sum(p(2:9) ./ denoms, 2);

t = zm1 + g + 0.5;
y_lanczos = sqrt(2*pi) .* t.^(zm1+0.5) .* exp(-t) .* x;

y = y_lanczos;
y(need_reflect) = pi ./ (sin(pi*z(need_reflect)) .* y_lanczos(need_reflect));
y = reshape(y, original_shape);
end


% =================================================================
% Compute orbit data for a given eccentric velocity η
% =================================================================
function orb = compute_orbit_data(eta, nPsi)
[rmin, rmax] = turning_points(eta);
T = find_radial_period(eta, rmin);
dt = T / nPsi;

r = rmin; u = 0.0; theta = 0.0;
X   = zeros(1,nPsi);
Yth = zeros(1,nPsi);
psi = 2*pi*(0:nPsi-1)/nPsi;

for j = 1:nPsi
    X(j)   = log(r);
    Yth(j) = theta;
    [r,u,theta] = rk4_step_all(r,u,theta,dt);
end

Omega_bar = theta / T;
tgrid     = (0:nPsi-1)*dt;
Y         = Yth - Omega_bar * tgrid;
kappa_bar = 2*pi / T;

I0 = T;
I1 = dt * sum(exp(-X));
I2 = dt * sum(exp(-2*X));

orb.eta   = eta;
orb.kappa = kappa_bar;
orb.Omega = Omega_bar;
orb.psi   = psi;
orb.X     = X;
orb.Y     = Y;
orb.I0    = I0;
orb.I1    = I1;
orb.I2    = I2;
end


function [rmin,rmax] = turning_points(eta)
f = @(r) eta.^2 + 1 - 2*log(r) - r.^(-2);
rmin = fzero(f, [0.001, 1.0]);
rmax = fzero(f, [1.0, 100.0]);
end


function T = find_radial_period(~, rmin)
dt = 0.01;
r = rmin; u = 0.0; theta = 0.0; t = 0.0;
last_u = u; crosses = 0;
for step = 1:200000
    [r,u,theta] = rk4_step_all(r,u,theta,dt);
    t_now = t + dt;
    if step > 1
        if last_u > 0 && u <= 0
            crosses = crosses + 1;
        elseif last_u < 0 && u >= 0
            crosses = crosses + 1;
        end
        if crosses == 2
            T = t_now;
            return;
        end
    end
    t = t_now;
    last_u = u;
end
error('Failed to find radial period');
end


function [r_new,u_new,theta_new] = rk4_step_all(r,u,theta,dt)
k1_r = u; k1_u = 1/r^3 - 1/r; k1_t = 1/r^2;
r2 = r + 0.5*dt*k1_r; u2 = u + 0.5*dt*k1_u;
k2_r = u2; k2_u = 1/r2^3 - 1/r2; k2_t = 1/r2^2;
r3 = r + 0.5*dt*k2_r; u3 = u + 0.5*dt*k2_u;
k3_r = u3; k3_u = 1/r3^3 - 1/r3; k3_t = 1/r3^2;
r4 = r + dt*k3_r; u4 = u + dt*k3_u;
k4_r = u4; k4_u = 1/r4^3 - 1/r4; k4_t = 1/r4^2;
r_new = r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r);
u_new = u + (dt/6)*(k1_u + 2*k2_u + 2*k3_u + k4_u);
theta_new = theta + (dt/6)*(k1_t + 2*k2_t + 2*k3_t + k4_t);
end


% =================================================================
% Compute Q_{ℓm}(α;η) - VECTORIZED
% =================================================================
function Q = compute_Q_lm(alpha, eta, orbitData, m, ellVec)
nAlpha = numel(alpha);
nEta   = numel(eta);
nEll   = numel(ellVec);
Q = zeros(nEll, nAlpha, nEta);

alpha_col = alpha(:);
ell_col   = ellVec(:);

for kEta = 1:nEta
    orb  = orbitData{kEta};
    X    = orb.X(:).';
    Y    = orb.Y(:).';
    psi  = orb.psi(:).';
    nPsi = numel(psi);
    dpsi = 2*pi / nPsi;
    
    base_kernel = exp(-0.5*X + 1i*m*Y);
    alpha_phase = exp(1i * alpha_col * X);
    kernel_alpha = alpha_phase .* base_kernel;
    ell_phase = exp(-1i * ell_col * psi);
    Q(:,:,kEta) = (dpsi/(2*pi)) * (ell_phase * kernel_alpha.');
end
end


% =================================================================
% Compute F_{ℓm} for ALL ℓ at once - VECTORIZED
% =================================================================
function F_all = compute_F_lm_all_ell(ellVec, kappa, Omega, a_par, m, Ncut, omega, h, Hhat_h, Hhat_prime_h, exp_h, exp_matrix, dh)
nEll = numel(ellVec);
ell_col = ellVec(:);

freq_term = ell_col * kappa + m * Omega;
numerator = ((a_par + 1) * freq_term - a_par * m) .* Hhat_h - m .* Hhat_prime_h;
denominator = freq_term - omega .* exp_h;
integrand_base = numerator ./ denominator;

F_all = (dh / (2*pi)) * (integrand_base * exp_matrix.');
end


% =================================================================
% Build kernel S_m(β,α) - OPTIMIZED
% =================================================================
function Sm = build_kernel(alpha, Kalpha, eta, wEta, orbitData, ...
                           Q_lmk, ellVec, a_par, m, Ncut, sigmaU, omega, Ca_tilde)

nAlpha = numel(alpha);
nEta   = numel(eta);
nEll   = numel(ellVec);

dAlpha = alpha(2) - alpha(1);
nuVals = (-nAlpha+1:nAlpha-1) * dAlpha;

[JJ, II] = meshgrid(1:nAlpha, 1:nAlpha);
idxMat = (II - JJ) + nAlpha;
idxMat_flat = idxMat(:);

I0_vec    = zeros(1, nEta);
kappa_vec = zeros(1, nEta);
Omega_vec = zeros(1, nEta);
for kEta = 1:nEta
    orb = orbitData{kEta};
    I0_vec(kEta)    = orb.I0;
    kappa_vec(kEta) = orb.kappa;
    Omega_vec(kEta) = orb.Omega;
end
gauss_eta_vec = exp(-eta.^2 / (2*sigmaU^2));
weight_vec    = Ca_tilde * I0_vec .* wEta .* eta .* gauss_eta_vec;

% Precompute h-grid quantities (DONE ONCE)
hmax = 12.0;
Nh   = 8192;
h    = linspace(-hmax, hmax, Nh);
dh   = h(2) - h(1);

Hhat_h       = 1 ./ (1 + exp(-Ncut * h));
Hhat_prime_h = Ncut * exp(-Ncut * h) ./ (1 + exp(-Ncut * h)).^2;
exp_h        = exp(h);

nuVals_col = nuVals(:);
h_row      = h(:).';
exp_matrix = exp(-1i * nuVals_col * h_row);

Sm = zeros(nAlpha, nAlpha);

for kEta = 1:nEta
    F_all = compute_F_lm_all_ell(ellVec, kappa_vec(kEta), Omega_vec(kEta), ...
                                 a_par, m, Ncut, omega, h, Hhat_h, Hhat_prime_h, exp_h, exp_matrix, dh);
    
    Sm_eta = zeros(nAlpha, nAlpha);
    for iEll = 1:nEll
        F_lm_vec = F_all(iEll, :).';
        F_lm_mat = reshape(F_lm_vec(idxMat_flat), nAlpha, nAlpha);
        Q_vec = Q_lmk(iEll, :, kEta);
        QQbar = (Q_vec.' * conj(Q_vec)).';
        Sm_eta = Sm_eta + QQbar .* F_lm_mat;
    end
    
    Sm = Sm + weight_vec(kEta) * Sm_eta;
end

Sm = Kalpha.' .* Sm;
end


% =================================================================
% FAST kernel builder using precomputed ω-independent data
% OPTIMIZED: Batches F_lm computation over all ℓ values per η
% =================================================================
function M = build_kernel_fast(omega, nAlpha, nEta, nEll, nNu, Nh, ...
                               freq_term_all, numerator_all, exp_h, exp_matrix, dh_val, ...
                               idxMat_flat, QQbar_flat, weight_vec, Kalpha, dAlpha)

Sm = zeros(nAlpha, nAlpha);
coeff = dh_val / (2*pi);

for kEta = 1:nEta
    % --- BATCH F_lm over all ℓ values ---
    % numerator_all is (Nh, nEll, nEta)
    num_batch = numerator_all(:,:,kEta);  % Nh x nEll
    
    % Compute all denominators at once: denom_batch(h, ℓ) = freq_term(ℓ) - omega * exp(h)
    denom_batch = freq_term_all(:,kEta).' - omega .* exp_h(:);  % Nh x nEll (broadcasting)
    
    % Integrand for all ℓ: (Nh x nEll)
    integrand_batch = num_batch ./ denom_batch;
    
    % Single matrix-matrix multiply: F_lm_batch (nNu x nEll)
    F_lm_batch = coeff * (exp_matrix * integrand_batch);
    
    % --- Accumulate Sm_eta using vectorized indexing ---
    % F_lm_batch(idxMat_flat, :) extracts the right values for each matrix position
    F_indexed = F_lm_batch(idxMat_flat, :);  % nAlpha^2 x nEll
    
    % Use precomputed flattened QQbar (already nAlpha^2 x nEll)
    QQ = QQbar_flat(:,:,kEta);  % nAlpha^2 x nEll
    
    % Element-wise multiply and sum over ℓ dimension
    Sm_eta_flat = sum(QQ .* F_indexed, 2);  % nAlpha^2 x 1
    Sm_eta = reshape(Sm_eta_flat, nAlpha, nAlpha);
    
    Sm = Sm + weight_vec(kEta) * Sm_eta;
end

% Apply K(α,m) and discretization
Sm = Kalpha.' .* Sm;
M = Sm * dAlpha;

end
