% ZANG_NEWTON_SEARCH
% ---------------------------------------------------------------
% Newton-Raphson search to find (Ω_p, s) such that λ(ω) = 1
% for the m=2 mode of the Zang disk.
%
% The eigenvalue condition for a mode is det(M - I) = 0, or
% equivalently, the largest eigenvalue λ = 1.
%
% We solve F(Ω_p, s) = 0 where F = [Re(λ) - 1; Im(λ)]
% using Newton's method with finite-difference Jacobians.
% ---------------------------------------------------------------

% --------------- Global model parameters -----------------------
m      = 2;          % angular harmonic
Ncut   = 4;          % cut-out index N
q      = 6;

sigmaU = 1/sqrt(q+1);   % radial velocity dispersion σ_u / V0  (Q≈1)
a_par  = q;  % 'a' in Zang's DF: a = (V0/σ_u)^2 - 1

% Wavenumber grid α (log-spiral basis)
nAlpha   = 101;       % must be odd; increase for better resolution
alphaMax = 20.0;     % |α| ≤ alphaMax; increase if needed
alpha    = linspace(-alphaMax, alphaMax, nAlpha);
% dAlpha   = alpha(2)-alpha(1);
dAlpha   = NewtonCotes(1, alpha);

% Eccentric-velocity grid η = U/V0
nEta   = 201;         % increase for better convergence
etaMax = 6.0*sigmaU;  % covers most of the Maxwellian

eta    = linspace(0, etaMax, nEta);
wEta   = NewtonCotes(2, eta);

% Radial Fourier harmonics ℓ range
ellMin = -100;
ellMax =  100;
ellVec = ellMin:ellMax;
nEll   = numel(ellVec);

% Number of orbital phase samples per radial period
NW_orbit = 16001;
NW       = 101;

% Build Kalnajs gravity factor K(α,m)
% Kalpha  = kalnajs_K(alpha, m);
Kalpha = kalnajs_Kc(alpha, m);  

% ---------- Precompute orbits and Fourier coefficients ----------
fprintf('Precomputing orbits and Q_{ℓm}(α;η)...\n');
orbitData = cell(1,nEta);
for k = 1:nEta
    orbitData{k} = compute_orbit_data(eta(k), NW_orbit, NW);
end

Q_lmk = compute_Q_lm(alpha, eta, orbitData, m, ellVec);

% ---------- Compute C_a tilde normalization (analytic, eq. 2.63) ----------
Ca_tilde_inv = sqrt(pi) * ((a_par+1)/2)^(-a_par/2) * gamma((a_par+1)/2) * exp((a_par+1)/2) / (a_par+1);
Ca_tilde     = 1.0 / Ca_tilde_inv ;

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

% =================================================================
% NEWTON-RAPHSON SEARCH FOR λ = 1
% =================================================================
fprintf('\n=== Newton search for mode eigenvalue (λ = 1) ===\n');

% Initial guess (Zang's values)
Omega_p = 0.439426;
s       = 0.127181;

% Newton parameters
max_iter = 50;
tol = 1e-12;
delta = 1e-5;  % finite difference step for Jacobian

fprintf('Initial: Ω_p = %.16f, s = %.6f\n', Omega_p, s);
fprintf('Target:  λ = 1 + 0i\n\n');

convergence_history = zeros(max_iter, 5);  % [Omega_p, s, Re(λ), Im(λ), |F|]

for iter = 1:max_iter
    omega = m*Omega_p + 1i*s;
    
    % Compute λ at current point
    [lambda, vec] = compute_eigenvalue(omega, precomp);
    
    % Residual: want λ = 1, so F = [Re(λ)-1; Im(λ)]
    F = [real(lambda) - 1; imag(lambda)];
    res_norm = norm(F);
    
    convergence_history(iter,:) = [Omega_p, s, real(lambda), imag(lambda), res_norm];
    
    fprintf('Iter %2d: Ω_p=%.10f  s=%.10f  λ=%.10f%+.10fi  |F|=%.2e\n', ...
            iter, Omega_p, s, real(lambda), imag(lambda), res_norm);
    
    % Check convergence
    if res_norm < tol
        fprintf('\n*** Converged! ***\n');
        convergence_history = convergence_history(1:iter,:);
        break;
    end
    
    % Compute Jacobian via finite differences
    % J = [dRe(λ)/dΩ_p, dRe(λ)/ds; dIm(λ)/dΩ_p, dIm(λ)/ds]
    
    % Partial w.r.t. Ω_p
    omega_dOp = m*(Omega_p + delta) + 1i*s;
    lambda_dOp = compute_eigenvalue(omega_dOp, precomp);
    dF_dOp = ([real(lambda_dOp); imag(lambda_dOp)] - [real(lambda); imag(lambda)]) / delta;
    
    % Partial w.r.t. s
    omega_ds = m*Omega_p + 1i*(s + delta);
    lambda_ds = compute_eigenvalue(omega_ds, precomp);
    dF_ds = ([real(lambda_ds); imag(lambda_ds)] - [real(lambda); imag(lambda)]) / delta;
    
    J = [dF_dOp, dF_ds];
    
    % Check Jacobian condition
    cond_J = cond(J);
    if cond_J > 1e10
        warning('Jacobian ill-conditioned (cond = %.2e), reducing step', cond_J);
    end
    
    % Newton step: x_new = x - J \ F
    dx = J \ F;
    
    % Line search (backtracking if needed)
    step = 1.0;
    for ls = 1:10
        Omega_p_new = Omega_p - step * dx(1);
        s_new = s - step * dx(2);
        
        % Ensure s > 0 (growth rate must be positive for unstable modes)
        if s_new > 1e-4 && Omega_p_new > 0 && Omega_p_new < 1
            omega_new = m*Omega_p_new + 1i*s_new;
            lambda_new = compute_eigenvalue(omega_new, precomp);
            F_new = [real(lambda_new) - 1; imag(lambda_new)];
            if norm(F_new) < res_norm || ls == 10
                break;
            end
        end
        step = step * 0.5;
    end
    
    if step < 1
        fprintf('         (backtracking: step = %.3f)\n', step);
    end
    
    Omega_p = Omega_p - step * dx(1);
    s = s - step * dx(2);
end

% Final result
omega = m*Omega_p + 1i*s;
[lambda_final, vec] = compute_eigenvalue(omega, precomp);

fprintf('\n=== Final Result ===\n');
fprintf('Ω_p = %.10f   (Zang: 0.439426)\n', Omega_p);
fprintf('s   = %.10f   (Zang: 0.127181)\n', s);
fprintf('λ   = %.10f + %.10fi\n', real(lambda_final), imag(lambda_final));
fprintf('|λ-1| = %.2e\n', abs(lambda_final - 1));

% Plot convergence
% figure('Name', 'Newton Convergence');
% subplot(2,1,1);
% semilogy(1:size(convergence_history,1), convergence_history(:,5), 'b.-', 'LineWidth', 1.5, 'MarkerSize', 15);
% xlabel('Iteration');
% ylabel('|F| = |[\Re(\lambda)-1, \Im(\lambda)]|');
% title('Newton Convergence');
% grid on;
% 
% subplot(2,1,2);
% plot(convergence_history(:,1), convergence_history(:,2), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
% hold on;
% plot(0.439426, 0.127181, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'k');
% xlabel('\Omega_p');
% ylabel('s');
% legend('Newton path', 'Zang reference');
% title('Search trajectory in (\Omega_p, s) space');
% grid on;


% =================================================================
% Compute largest eigenvalue of M(ω) = S_m * dα
% =================================================================
function [lambda, vec] = compute_eigenvalue(omega, precomp)

Sm = build_kernel(precomp.alpha, precomp.Kalpha, precomp.eta, precomp.wEta, ...
                  precomp.orbitData, precomp.Q_lmk, precomp.ellVec, ...
                  precomp.a_par, precomp.m, precomp.Ncut, precomp.sigmaU, omega, ...
                  precomp.Ca_tilde);

M = Sm .* repmat(precomp.dAlpha, length(precomp.dAlpha), 1);

% Find eigenvalue with largest real part (for mode finding)
[V, D] = eig(M);
eigvals = diag(D);
[~, idx] = max(real(eigvals));
lambda = eigvals(idx);
vec = V(:, idx);
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
K = real(K)/2;
end

function K = kalnajs_Kc(alpha, m)
alpha = alpha(:).';       % row vector
ia    = 1i*alpha;

z1 = (m + 0.5 + ia)/2;
z2 = (m + 0.5 - ia)/2;
z3 = (m + 1.5 + ia)/2;
z4 = (m + 1.5 - ia)/2;

K = gamma_complex(z1).*gamma_complex(z2) ./ (gamma_complex(z3).*gamma_complex(z4));
K = real(K)/2;              % should be purely real & positive
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
function orb = compute_orbit_data(eta, NW_orbit, NW);
    
Omega_bar = 1;
kappa_bar = sqrt(2.0);
I0 = 2*pi/kappa_bar;
I1 = I0;
I2 = I0;

w = linspace(0, pi, NW_orbit);
d_w = w(2) - w(1);
cos_w = cos(w);
sin_w = sin(w);

if eta < 1e-10
    phi = w * Omega_bar/kappa_bar;
    xs = 1 - 0*cos_w;
    t = w/kappa_bar;
    X = log(xs);
    Y = phi - Omega_bar * t;
    psi = w;

elseif eta < 0.01
    a = eta/sqrt(2);
    xs = 1 - a*cos_w;
    phi = w*Omega_bar/kappa_bar + 2*Omega_bar/kappa_bar*a*sin_w;
    t = w/kappa_bar;
    X = log(xs);
    Y = phi - Omega_bar * t;
    psi = w;

else
    [rmin, rmax] = turning_points(eta);
    rs = (rmax+rmin)/2;
    ds = (rmax-rmin)/2;

    xs = rs - ds*cos_w;
    rvr = sqrt(xs.^2 .* (eta.^2 + 1 - 2*log(xs)) - 1);

    svr = sin_w.*xs./rvr;
    svr(1)   = svr(2)*2     - svr(3);
    svr(end) = svr(end-1)*2 - svr(end-2);

    dt1 = ds * d_w .* svr;
    dt2 = zeros(1, NW_orbit);
    dt2(2:end) = (dt1(1:end-1)+dt1(2:end))/2;
    t_half = cumsum(dt2);
    T_half = t_half(end);
    kappa_bar = pi/T_half;
    X = log(xs);
    psi = kappa_bar*t_half;

    dphi1 = dt1./xs.^2;
    dphi2 = zeros(1, NW_orbit);
    dphi2(2:end) = (dphi1(1:end-1)+dphi1(2:end))/2;
    phi_half = cumsum(dphi2);
    Omega_bar = phi_half(end)/T_half;
    Y = phi_half - Omega_bar * t_half;

    I0 = 2*T_half;
    I1 = 2*sum(dt1./xs);
    I2 = 2*phi_half(end);

end

Step = (NW_orbit-1)/(NW-1);
IS = 1:Step:NW_orbit;

orb.eta   = eta;
orb.kappa = kappa_bar;
orb.Omega = Omega_bar;
orb.I0    = I0;
orb.I1    = I1;
orb.I2    = I2;

orb.psi   = psi(IS);
orb.X     = X(IS);
orb.Y     = Y(IS);

end


function [rmin,rmax] = turning_points(eta)
f = @(r) eta.^2 + 1 - 2*log(r) - r.^(-2);
rmin = fzero(f, [0.1, 1.0]);
rmax = fzero(f, [1.0, 2*exp((eta^2+1)/2)]);
end


% =================================================================
% Compute Q_{ℓm}(α;η) - VECTORIZED
% =================================================================
function Q = compute_Q_lm(alpha, eta, orbitData, m, ellVec)
nAlpha = numel(alpha);
nEta   = numel(eta);
nEll   = numel(ellVec);
Q = zeros(nEll, nAlpha, nEta);

ell_col   = ellVec(:);

for kEta = 1:nEta
    orb  = orbitData{kEta};
    X    = orb.X(:).';
    Y    = orb.Y(:).';
    psi  = orb.psi(:).';
    dpsi = TrapNE_Coef(psi);
    for ia = 1:nAlpha,
        alpha_i = alpha(ia);
        base_kernel = exp((1i * alpha_i-0.5)*X ) .* dpsi;
        ell_phase = cos(ell_col * psi - m*repmat(Y,length(ell_col),1));
        Q(:,ia,kEta) = (ell_phase * base_kernel.')/pi;
    end
end
end


% =================================================================
% Compute F_{ℓm} for ALL ℓ at once - VECTORIZED
% =================================================================
% function F_all = compute_F_lm_all_ell(ellVec, kappa, Omega, a_par, m, Ncut, omega, h, Hhat_h, Hhat_prime_h, exp_h, exp_matrix)
function F_all = compute_F_lm_all_ell(ellVec, kappa, Omega, a_par, m, omega, Hhat_h, Hhat_prime_h, exp_h, exp_matrix)
%nEll = numel(ellVec);
ell_col = ellVec(:);

freq_term = ell_col * kappa + m * Omega;
numerator = ((a_par + 1) * freq_term - a_par * m) .* Hhat_h - m .* Hhat_prime_h;
denominator = freq_term - omega .* exp_h;
integrand_base = numerator ./ denominator;

F_all = (integrand_base * exp_matrix.') / (2*pi);
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

% Precompute h-grid quantities
hmax = 4.0;
Nh   = 8192/2;
h    = linspace(-hmax, hmax, Nh+1);
% dh   = h(2) - h(1);
dh = NewtonCotes(2, h);
fprintf('Nh+1 = %d (should be odd for Simpson), length(h) = %d, length(dh) = %d\n', Nh+1, length(h), length(dh));
fprintf('Sum of weights: %.10f, expected: %.10f\n', sum(dh), 2*hmax);

Hhat_h       = 1 ./ (1 + exp(-Ncut * h));
Hhat_prime_h = Ncut * exp(-Ncut * h) ./ (1 + exp(-Ncut * h)).^2;
exp_h        = exp(h);

nuVals_col = nuVals(:);
h_row      = h(:).';
exp_matrix = exp(-1i * nuVals_col * h_row) .* repmat(dh, length(nuVals_col), 1 );

Sm = zeros(nAlpha, nAlpha);
trace_Sm_eta = zeros(1, nEta);  % Diagnostic: store trace for each eta

for kEta = 1:nEta
    F_all = compute_F_lm_all_ell(ellVec, kappa_vec(kEta), Omega_vec(kEta), ...
                                 a_par, m, omega, Hhat_h, Hhat_prime_h, exp_h, exp_matrix);
    
    Sm_eta = zeros(nAlpha, nAlpha);
    for iEll = 1:nEll
        F_lm_vec = F_all(iEll, :).';
        F_lm_mat = reshape(F_lm_vec(idxMat_flat), nAlpha, nAlpha);
        Q_vec = Q_lmk(iEll, :, kEta);
        QQbar = (Q_vec.' * conj(Q_vec)).';
        Sm_eta = Sm_eta + QQbar .* F_lm_mat;
    end
    
    trace_Sm_eta(kEta) = trace(Sm_eta);  % Diagnostic: store trace
    Sm = Sm + weight_vec(kEta) * Sm_eta;
end

% --- Diagnostic plot ---
f_eta = I0_vec .* eta .* gauss_eta_vec .* trace_Sm_eta;
figure;
subplot(2,1,1);
plot(double(eta), double(real(f_eta)), 'b.-', double(eta), double(imag(f_eta)), 'r.-');
xlabel('\eta'); ylabel('f(\eta)'); legend('Re','Im');
title('f(\eta) = I_0 \cdot \eta \cdot e^{-\eta^2/2\sigma^2} \cdot tr(S_m^\eta)');
grid on;
subplot(2,1,2);
plot(double(eta(2:end)), double(diff(real(f_eta))), 'b.-');
xlabel('\eta'); ylabel('\Delta Re(f)');
title('Finite difference - look for rapid oscillations');
grid on;
% --- End diagnostic ---

Sm = Kalpha.' .* Sm;
end
