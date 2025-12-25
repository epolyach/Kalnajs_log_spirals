function zang_logspiral
% ZANG_LOGSPIRAL_N4
% ---------------------------------------------------------------
% Matrix method for non‑axisymmetric modes of a hot Mestel disk,
% following Zang (1976) using a logarithmic‑spiral expansion.
%
% This script targets the m=2, N=4 cut‑out disk at Q=1 and lets
% you evaluate the largest eigenvalue lambda(omega) of the
% integral equation
%     A(β) = ∫ S_m(β,α; σ_u, ω) A(α) dα
% discretized as a matrix problem  A = S(ω) A.
%
% The quoted Zang numbers for this case (p.134) are
%   growth rate  s = 0.1270
%   pattern speed Ω_p = 0.4394
% so that ω = m Ω_p + i s = 2*Ω_p + i s.
%
% NOTES
%  * Units: r0 = 1, V0 = 1, GΣ0 absorbed into an overall factor.
%  * Orbits in the 1/r Mestel potential are integrated numerically.
%  * The angular‑momentum integral is evaluated numerically via
%    a Fourier transform of the cut‑out function H(J).
%  * The code is intentionally modular so you can refine each
%    piece (orbit resolution, number of harmonics, α‑grid, etc.)
%    until it converges for your purposes.
%
% EXPECTATIONS
%  * Out of the box, with the coarse defaults below, you should
%    get a largest eigenvalue |lambda| of order unity for
%    (Ω_p,s) near Zang’s values, but not identical numbers.
%  * To get to ~1e‑3 accuracy you will need to:
%      - tighten the α‑grid (nAlpha, alphaMax),
%      - increase nEta, nPsi, and the ℓ‑range,
%      - possibly refine the Ĥ(ν) integral.
% ---------------------------------------------------------------

% --------------- Global model parameters -----------------------
m      = 2;          % angular harmonic
Ncut   = 4;          % cut‑out index N
q      = 6;

sigmaU = 1/sqrt(q+1);   % radial velocity dispersion σ_u / V0  (Q≈1)
a_par  = q;  % 'a' in Zang's DF: a = (V0/σ_u)^2 - 1

% Wavenumber grid α (log‑spiral basis)
nAlpha   = 301;       % must be odd; increase for better resolution
alphaMax = 10.0;     % |α| ≤ alphaMax; increase if needed
alpha    = linspace(-alphaMax, alphaMax, nAlpha);
% dAlpha   = alpha(2)-alpha(1);
dAlpha   = NewtonCotes(1, alpha);

% Eccentric‑velocity grid η = U/V0
nEta   = 100;         % increase for better convergence
etaMax = 5.0*sigmaU; % covers most of the Maxwellian

eta    = linspace(1e-6, etaMax, nEta);
% wEta   = eta_spacing_weights(eta);  % simple trapezoid weights
wEta   = NewtonCotes(2, eta);

% eta = etaMax*logspace(-6,0,nEta);
% u = log(eta);
% wEta   = eta .* NewtonCotes(2, u);

% Radial Fourier harmonics ℓ range
ellMin = -10;         % Zang often uses bigger ranges; start modest
ellMax =  10;
ellVec = ellMin:ellMax;
nEll   = numel(ellVec);

% Number of orbital phase samples per radial period
nPsi   = 256;         % increase e.g. to 128 or 256 for more accuracy

% Build Kalnajs gravity factor K(α,m)
Kalpha  = kalnajs_K(alpha, m);  % 1 x nAlpha, real & >0
% Kalpha1 = kalnajs_Kc(alpha, m);  

% ---------- Precompute orbits and Fourier coefficients ----------
fprintf('Precomputing orbits and Q_{ℓm}(α;η)...\n');
orbitData = cell(1,nEta);
for k = 1:nEta
    orbitData{k} = compute_orbit_data(eta(k), nPsi);
end

Q_lmk = compute_Q_lm(alpha, eta, orbitData, m, ellVec);

% ---------- Compute C_a tilde normalization ----------
% Method 1: Analytic formula (Zang eq. 2.63)
%   C_a_tilde^{-1} = sqrt(π) [(a+1)/2]^{a/2} Γ[(a+1)/2] e^{(a+1)/2} / (a+1)
Ca_tilde_inv_263 = sqrt(pi) * ((a_par+1)/2)^(-a_par/2) * gamma((a_par+1)/2) * exp((a_par+1)/2) / (a_par+1);
Ca_tilde_263     = 1.0 / Ca_tilde_inv_263;

% Method 2: Numerical integration (Zang eq. 2.64)
%   C_a_tilde^{-1} = ∫_0^∞ U I_1(U) exp[-(a+1) U^2 / 2] dU
I1_list = zeros(1, nEta);
for k = 1:nEta
    I1_list(k) = orbitData{k}.I1;   % I_1(η_k) per eq. (2.39)
end
Ca_tilde_inv_264 = sum(eta .* I1_list .* exp(-0.5*(a_par+1)*eta.^2) .* wEta);
Ca_tilde_264     = 1.0 / Ca_tilde_inv_264;

% Compare both methods
fprintf('\n--- C_a tilde normalization comparison ---\n');
fprintf('Eq. (2.63) analytic:    Ca_tilde = %.6e  (Ca_tilde_inv = %.6e)\n', Ca_tilde_263, Ca_tilde_inv_263);
fprintf('Eq. (2.64) numerical:   Ca_tilde = %.6e  (Ca_tilde_inv = %.6e)\n', Ca_tilde_264, Ca_tilde_inv_264);
fprintf('Relative difference:    %.4e\n', abs(Ca_tilde_263 - Ca_tilde_264) / Ca_tilde_263);

% Use the analytic value (more accurate)
Ca_tilde = Ca_tilde_263/2;

% ----------------- Pack precomputed data into struct ----------------
precomp.alpha    = alpha;
precomp.dAlpha   = dAlpha;
precomp.Kalpha   = Kalpha;
precomp.eta      = eta;
precomp.wEta     = wEta;
precomp.orbitData = orbitData;
precomp.Q_lmk    = Q_lmk;
precomp.ellVec   = ellVec;
precomp.a_par    = a_par;
precomp.m        = m;
precomp.Ncut     = Ncut;
precomp.sigmaU   = sigmaU;
precomp.Ca_tilde = Ca_tilde;

% ----------------- Diagnostic: eigenvalue spectrum at Zang's values ------
Omega_p = 0.439426;
s       = 0.127181;
omega   = m*Omega_p + 1i*s;

fprintf('\n--- Diagnostic: Eigenvalue spectrum at Zang values ---\n');
fprintf('ω = %.4f + i*%.4f  (Ω_p = %.4f, s = %.4f)\n', real(omega), imag(omega), Omega_p, s);

% Build kernel and get multiple eigenvalues
Sm = build_kernel(alpha, Kalpha, eta, wEta, orbitData, Q_lmk, ...
                  ellVec, a_par, m, Ncut, sigmaU, omega, Ca_tilde);
M = Sm * dAlpha;

fprintf('\nMatrix M size: %d x %d\n', size(M,1), size(M,2));
fprintf('Matrix M norm: %.4e\n', norm(M, 'fro'));

% Get all eigenvalues and sort by imaginary part (descending)
[V, D] = eig(M);
eigvals_all = diag(D);
[~, idx] = sort(real(eigvals_all), 'descend');
eigvals = eigvals_all(idx(1:6));  % top 6 by Re(λ)
plot(real(eigvals_all), imag(eigvals_all), '.')
V = V(:, idx);

fprintf('\nTop 6 eigenvalues of M(ω):\n');
fprintf('  #  |   Re(λ)   |   Im(λ)   |   |λ|\n');
fprintf('-----+----------+----------+--------\n');
for k = 1:6
    fprintf('  %d  | %8.5f | %8.5f | %6.4f\n', k, real(eigvals(k)), imag(eigvals(k)), abs(eigvals(k)));
end

fprintf('\nFor a mode, need λ = 1 (Re=1, Im=0)\n');

vec = V(:,1);
lambda_final = eigvals(1);

% --------------- Plot the dominant eigenfunction A(α) ------------
% A_alpha = vec;
% 
% figure;
% subplot(2,1,1);
% plot(alpha, real(A_alpha), 'b-', alpha, imag(A_alpha), 'r--', 'LineWidth',1.5);
% xlabel('\alpha'); ylabel('A(\alpha)');
% legend('Re','Im');
% title(sprintf('Dominant eigenfunction A(\\alpha) at \\Omega_p=%.4f, s=%.4f', ...
%               Omega_p, s));
% 
% subplot(2,1,2);
% plot(alpha, abs(A_alpha), 'k-', 'LineWidth',1.5);
% xlabel('\alpha'); ylabel('|A(\alpha)|');
% title('|A(\alpha)|');

end % zang_logspiral_N4


% =================================================================
% Residual function for fsolve: returns [Re(λ)-1; Im(λ)]
% =================================================================
function res = lambda_residual(x, precomp)
% x = [Ω_p; s]
Omega_p = x(1);
s       = x(2);
m       = precomp.m;
omega   = m*Omega_p + 1i*s;

lambda = compute_eigenvalue(omega, precomp);

res = [real(lambda) - 1; imag(lambda)];
end


% =================================================================
% Compute largest eigenvalue of M(ω) = S_m * dα
% =================================================================
function [lambda, vec] = compute_eigenvalue(omega, precomp)

Sm = build_kernel(precomp.alpha, precomp.Kalpha, precomp.eta, precomp.wEta, ...
                  precomp.orbitData, precomp.Q_lmk, precomp.ellVec, ...
                  precomp.a_par, precomp.m, precomp.Ncut, precomp.sigmaU, omega, ...
                  precomp.Ca_tilde);

M = Sm * precomp.dAlpha;

% Find eigenvalue with largest imaginary part
[V, D] = eig(M);
eigvals = diag(D);
[~, idx] = max(imag(eigvals));
lambda = eigvals(idx);
vec = V(:, idx);
end


% =================================================================
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
alpha = alpha(:).';       % row vector
ia    = 1i*alpha;

z1 = (m + 0.5 + ia)/2;
z2 = (m + 0.5 - ia)/2;
z3 = (m + 1.5 + ia)/2;
z4 = (m + 1.5 - ia)/2;

K = cgamma(z1).*cgamma(z2) ./ (cgamma(z3).*cgamma(z4));
K = real(K);              % should be purely real & positive
end

function K = kalnajs_Kc(alpha, m)
alpha = alpha(:).';       % row vector
ia    = 1i*alpha;

z1 = (m + 0.5 + ia)/2;
z2 = (m + 0.5 - ia)/2;
z3 = (m + 1.5 + ia)/2;
z4 = (m + 1.5 - ia)/2;

K = gamma_complex(z1).*gamma_complex(z2) ./ (gamma_complex(z3).*gamma_complex(z4));
K = real(K);              % should be purely real & positive
end

% =================================================================
% Complex gamma function using Lanczos approximation
% VECTORIZED version
% =================================================================
function y = cgamma(z)
% CGAMMA  Gamma function for complex arguments using Lanczos approximation.
%   Works for Re(z) > 0; uses reflection formula for Re(z) <= 0.
%   Fully vectorized.

g = 7;
p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, ...
     771.32342877765313, -176.61502916214059, 12.507343278686905, ...
     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

original_shape = size(z);
z = z(:);  % flatten
n = numel(z);

% Identify which elements need reflection
need_reflect = real(z) < 0.5;

% For reflection: compute Gamma(1-z) then apply formula
z_work = z;
z_work(need_reflect) = 1 - z(need_reflect);

% Lanczos approximation for all elements (vectorized)
zm1 = z_work - 1;

% Compute x = p(1) + sum(p(i)/(z + i - 1)) for i=2..9
% Build matrix: (n x 8) where each row is [1./(zm1+1), 1./(zm1+2), ..., 1./(zm1+8)]
denoms = zm1 + (1:8);  % n x 8 via broadcasting
x = p(1) + sum(p(2:9) ./ denoms, 2);  % n x 1

t = zm1 + g + 0.5;
y_lanczos = sqrt(2*pi) .* t.^(zm1+0.5) .* exp(-t) .* x;

% Apply reflection formula where needed
y = y_lanczos;
y(need_reflect) = pi ./ (sin(pi*z(need_reflect)) .* y_lanczos(need_reflect));

% Reshape to original
y = reshape(y, original_shape);
end


% =================================================================
% Compute orbit data for a given eccentric velocity η
% in the Mestel potential  Φ = ln r  (V0=1, r0=1)
%
% Returns a struct with fields:
%   eta, kappa, Omega, psi[1..nPsi], X[ψ]=ln r, Y[ψ]=θ - Ω t
% =================================================================
function orb = compute_orbit_data(eta, nPsi)

% 1) Find turning points r_min, r_max from  U^2 + 1 - 2 ln r - r^{-2} = 0
[rmin, rmax] = turning_points(eta);

% 2) Find radial period T by a coarse integration (this is I0(η))
T = find_radial_period(eta, rmin);

% 3) Re‑integrate with uniform phase sampling
dt = T / nPsi;

r     = rmin;
u     = 0.0;
theta = 0.0;
t     = 0.0;

X   = zeros(1,nPsi);
Yth = zeros(1,nPsi);
psi = 2*pi*(0:nPsi-1)/nPsi;

for j = 1:nPsi
    X(j)   = log(r);
    Yth(j) = theta;
    % advance one RK4 step
    [r,u,theta] = rk4_step_all(r,u,theta,dt);
    t = t + dt;
end

Omega_bar = theta / T;
tgrid     = (0:nPsi-1)*dt;
Y         = Yth - Omega_bar * tgrid;

kappa_bar = 2*pi / T;

% Compute I_n integrals via eq. (2.39):
%   I_n = 2 ∫ dx / [x^n sqrt(...)] = ∫ r^{-n} dt  (over one period)
% With uniform time sampling: I_n = (T/nPsi) * sum(r^{-n})
I0 = T;                                      % n=0: radial period
I1 = dt * sum(exp(-X));                      % n=1: ∫ (1/r) dt
I2 = dt * sum(exp(-2*X));                    % n=2: ∫ (1/r^2) dt = Θ(r_H) per (2.40)

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


% =================================================================
% Root finder for turning points r_min, r_max.
% Solve f(r) = η^2 + 1 - 2 ln r - r^{-2} = 0  for r<1 and r>1.
% =================================================================
function [rmin,rmax] = turning_points(eta)

f = @(r) eta.^2 + 1 - 2*log(r) - r.^(-2);

% left root: start between 0.2 and 1
rmin = fzero(f, [0.001, 1.0]);

% right root: start between 1 and, say, 5
rmax = fzero(f, [1.0, 100.0]);

end


% =================================================================
% Coarse integration to estimate radial period T(η).
% Integrate r'' = 1/r^3 - 1/r starting from pericenter r=rmin,u=0.
% Detect two sign changes of u: +→- (apocenter), -→+ (next pericenter).
% =================================================================
function T = find_radial_period(~, rmin)

dt      = 0.01;      % coarse step
r       = rmin;
u       = 0.0;
theta   = 0.0;       %#ok<NASGU>
t       = 0.0;
last_u  = u;
crosses = 0;
maxSteps = 200000;

for step = 1:maxSteps
    [r,u,theta] = rk4_step_all(r,u,theta,dt); %#ok<ASGLU>
    t_now = t + dt;
    % detect sign changes of u
    if step > 1
        if last_u > 0 && u <= 0
            crosses = crosses + 1;  % apocenter
        elseif last_u < 0 && u >= 0
            crosses = crosses + 1;  % pericenter
        end
        if crosses == 2
            T = t_now;
            return;
        end
    end
    t      = t_now;
    last_u = u;
end

error('Failed to find radial period; increase maxSteps or reduce dt.');
end


% =================================================================
% One RK4 step for (r,u,theta) system:
%   dr/dt = u
%   du/dt = 1/r^3 - 1/r
%   dθ/dt = 1/r^2
% =================================================================
function [r_new,u_new,theta_new] = rk4_step_all(r,u,theta,dt)

k1_r = u;
k1_u = 1/r^3 - 1/r;
k1_t = 1/r^2;

r2   = r + 0.5*dt*k1_r;
u2   = u + 0.5*dt*k1_u;
k2_r = u2;
k2_u = 1/r2^3 - 1/r2;
k2_t = 1/r2^2;

r3   = r + 0.5*dt*k2_r;
u3   = u + 0.5*dt*k2_u;
k3_r = u3;
k3_u = 1/r3^3 - 1/r3;
k3_t = 1/r3^2;

r4   = r + dt*k3_r;
u4   = u + dt*k3_u;
k4_r = u4;
k4_u = 1/r4^3 - 1/r4;
k4_t = 1/r4^2;

r_new     = r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r);
u_new     = u + (dt/6)*(k1_u + 2*k2_u + 2*k3_u + k4_u);
theta_new = theta + (dt/6)*(k1_t + 2*k2_t + 2*k3_t + k4_t);
end


% =================================================================
% Compute Q_{ℓm}(α;η) for all ℓ, α, η using eq. (3.27):
%   Q_{ℓm}(α;η) = (1/2π) ∫_0^{2π}
%       exp[(iα - 1/2) X(ψ;η) + i m Y(ψ;η) - i ℓ ψ] dψ
%
% Returns Q as a 3‑D array: Q(ellIndex, alphaIndex, etaIndex).
% VECTORIZED version - no loops over α or ℓ
% =================================================================
function Q = compute_Q_lm(alpha, eta, orbitData, m, ellVec)

nAlpha = numel(alpha);
nEta   = numel(eta);
nEll   = numel(ellVec);

Q = zeros(nEll, nAlpha, nEta);

% Reshape for broadcasting
alpha_col = alpha(:);          % nAlpha x 1
ell_col   = ellVec(:);         % nEll x 1

for kEta = 1:nEta
    orb  = orbitData{kEta};
    X    = orb.X(:).';         % 1 x nPsi (row)
    Y    = orb.Y(:).';         % 1 x nPsi (row)
    psi  = orb.psi(:).';       % 1 x nPsi (row)
    nPsi = numel(psi);
    dpsi = 2*pi / nPsi;
    
    % Compute base kernel: exp(-0.5*X + i*m*Y) for all ψ
    base_kernel = exp(-0.5*X + 1i*m*Y);   % 1 x nPsi
    
    % Phase factors for all α: exp(i*α*X) → nAlpha x nPsi
    alpha_phase = exp(1i * alpha_col * X);  % nAlpha x nPsi
    
    % Combined kernel for all α: nAlpha x nPsi
    kernel_alpha = alpha_phase .* base_kernel;  % broadcasts base_kernel
    
    % Phase factors for all ℓ: exp(-i*ℓ*ψ) → nEll x nPsi
    ell_phase = exp(-1i * ell_col * psi);  % nEll x nPsi
    
    % For each (ℓ, α), integrate over ψ:
    % Q(ℓ,α) = (1/2π) * dpsi * sum over ψ of [kernel_alpha(α,ψ) * ell_phase(ℓ,ψ)]
    % This is a matrix product: (nEll x nPsi) * (nPsi x nAlpha) = nEll x nAlpha
    Q(:,:,kEta) = (dpsi/(2*pi)) * (ell_phase * kernel_alpha.');
end

end


% =================================================================
% Compute F_{lm}(ν; η, ω) for ALL ℓ values at once via eq (3.41):
%   F_{lm}(ν) = (1/2π) ∫ [(a+1)(ℓκ̃+mΩ̃) - am]Ĥ(h) - mĤ'(h)
%                        / [ℓκ̃ + mΩ̃ - ω̃ e^h] × e^{-iνh} dh
% where Ĥ(h) = 1/(1 + e^{-Nh}) and Ĥ'(h) = dĤ/dh
%
% FULLY VECTORIZED: computes F(iℓ, iν) for all ℓ and ν at once
% Input: ellVec (1 x nEll), kappa, Omega scalars for this η
% Output: F_all (nEll x nNu)
% =================================================================
function F_all = compute_F_lm_all_ell(ellVec, kappa, Omega, a_par, m, Ncut, omega, h, Hhat_h, Hhat_prime_h, exp_h, exp_matrix, dh)

nEll = numel(ellVec);
ell_col = ellVec(:);  % nEll x 1

% freq_term(ℓ) = ℓκ + mΩ for each ℓ
freq_term = ell_col * kappa + m * Omega;  % nEll x 1

% Numerator: [(a+1)*freq_term - am] * Ĥ(h) - m * Ĥ'(h)
% Shape: (nEll x 1) .* (1 x Nh) = nEll x Nh
numerator = ((a_par + 1) * freq_term - a_par * m) .* Hhat_h - m .* Hhat_prime_h;

% Denominator: freq_term - ω * exp(h)
% Shape: (nEll x 1) - (1 x Nh) = nEll x Nh
denominator = freq_term - omega .* exp_h;

% Integrand base: nEll x Nh
integrand_base = numerator ./ denominator;

% Integration: F(ℓ, ν) = (dh/2π) * sum_h [integrand(ℓ,h) * exp(-iνh)]
% integrand_base: nEll x Nh
% exp_matrix: nNu x Nh
% We need: F(iℓ, iν) = integrand_base(iℓ, :) * exp_matrix(iν, :)'
% This is: integrand_base @ exp_matrix' = (nEll x Nh) @ (Nh x nNu) = nEll x nNu
F_all = (dh / (2*pi)) * (integrand_base * exp_matrix.');

end


% =================================================================
% Build kernel S_m(β,α; σ_u, ω) using eq. (3.40) with F_{ℓm} from (3.41)
%
%   S_m(β,α) = K(α,m) * ∑_ℓ ∫ dη W(η) Q_{ℓm}(α;η) Q̄_{ℓm}(β;η)
%                                    F_{ℓm}(β-α;η,ω) η dη
%
% where W(η) includes the Gaussian e^{-η^2/2σ_u^2}.
% HIGHLY OPTIMIZED: precompute exp_matrix, vectorize over all ℓ
% =================================================================
function Sm = build_kernel(alpha, Kalpha, eta, wEta, orbitData, ...
                           Q_lmk, ellVec, a_par, m, Ncut, sigmaU, omega, Ca_tilde)

nAlpha = numel(alpha);
nEta   = numel(eta);
nEll   = numel(ellVec);

% Precompute unique ν values and index mapping (done once)
dAlpha = alpha(2) - alpha(1);
nuVals = (-nAlpha+1:nAlpha-1) * dAlpha;  % All possible ν = β - α values
nNu    = numel(nuVals);

% Build index matrix for F_lm lookup: nuMat(i,j) corresponds to β_i - α_j
% Index into nuVals: idx = (i - j) + nAlpha  (1-based)
[JJ, II] = meshgrid(1:nAlpha, 1:nAlpha);
idxMat = (II - JJ) + nAlpha;  % nAlpha x nAlpha, values in 1..2*nAlpha-1
idxMat_flat = idxMat(:);  % Flatten once

% Precompute η-dependent quantities
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
weight_vec    = Ca_tilde * I0_vec .* wEta .* eta .* gauss_eta_vec;  % 1 x nEta

% Precompute h-grid quantities for F_lm (DONE ONCE)
hmax = 12.0;
Nh   = 8192;
h    = linspace(-hmax, hmax, Nh);
dh   = h(2) - h(1);

% These depend only on Ncut and h (not on ℓ, η, or ν)
Hhat_h       = 1 ./ (1 + exp(-Ncut * h));  % 1 x Nh
Hhat_prime_h = Ncut * exp(-Ncut * h) ./ (1 + exp(-Ncut * h)).^2;  % 1 x Nh
exp_h        = exp(h);  % 1 x Nh

% Precompute exp(-i ν h) matrix for all ν values (DONE ONCE - this is expensive!)
nuVals_col = nuVals(:);  % nNu x 1
h_row      = h(:).';     % 1 x Nh
exp_matrix = exp(-1i * nuVals_col * h_row);  % nNu x Nh

Sm = zeros(nAlpha, nAlpha);  % (i,j) = β_i, α_j

% Loop over η only (vectorize over ℓ)
for kEta = 1:nEta
    % Compute F_{ℓm}(ν) for ALL ℓ values at once
    F_all = compute_F_lm_all_ell(ellVec, kappa_vec(kEta), Omega_vec(kEta), ...
                                 a_par, m, Ncut, omega, h, Hhat_h, Hhat_prime_h, exp_h, exp_matrix, dh);
    % F_all is nEll x nNu
    
    % Sum over ℓ for this η
    Sm_eta = zeros(nAlpha, nAlpha);
    
    for iEll = 1:nEll
        % Map F to matrix form
        F_lm_vec = F_all(iEll, :).';  % nNu x 1
        F_lm_mat = reshape(F_lm_vec(idxMat_flat), nAlpha, nAlpha);
        
        % Q_{ℓm}(α;η_k)
        Q_vec = Q_lmk(iEll, :, kEta);  % 1 x nAlpha
        
        % Outer product Q(α_j) * conj(Q(β_i))
        QQbar = (Q_vec.' * conj(Q_vec)).';  % nAlpha x nAlpha
        
        Sm_eta = Sm_eta + QQbar .* F_lm_mat;
    end
    
    % Weight and accumulate
    Sm = Sm + weight_vec(kEta) * Sm_eta;
end

% Apply K(α,m) factor (column-wise, depends only on α)
Sm = Kalpha.' .* Sm;

end
