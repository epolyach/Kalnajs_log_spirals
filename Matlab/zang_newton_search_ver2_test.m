% ZANG_NEWTON_SEARCH_VER2_TEST
% ---------------------------------------------------------------
% Test: compute λ at FIXED ω for different nEta values
% to see if Re(λ) converges while Im(λ) oscillates.
% ---------------------------------------------------------------

% --------------- Global model parameters -----------------------
m      = 2;          % angular harmonic
Ncut   = 4;          % cut-out index N
q      = 6;

sigmaU = 1/sqrt(q+1);   % radial velocity dispersion σ_u / V0  (Q≈1)
a_par  = q;  % 'a' in Zang's DF: a = (V0/σ_u)^2 - 1

% Wavenumber grid α (log-spiral basis)
nAlpha   = 301;       % must be odd; increase for better resolution
alphaMax = 30.0;     % |α| ≤ alphaMax; increase if needed
alpha    = linspace(-alphaMax, alphaMax, nAlpha);
dAlpha   = NewtonCotes(1, alpha);

% Radial Fourier harmonics ℓ range
ellMin = -100;
ellMax =  100;
ellVec = ellMin:ellMax;
nEll   = numel(ellVec);

% Number of orbital phase samples per radial period
NW_orbit = 16001;
NW       = 401;

% Build Kalnajs gravity factor K(α,m)
Kalpha = kalnajs_Kc(alpha, m);  

% Fixed omega for testing (near converged value)
Omega_p_fixed = 0.4394429;
s_fixed       = 0.1272015;
omega_fixed = m*Omega_p_fixed + 1i*s_fixed;

fprintf('=== Testing λ at fixed ω = %.6f + %.6fi ===\n', real(omega_fixed), imag(omega_fixed));
fprintf('    (Ω_p = %.7f, s = %.7f)\n\n', Omega_p_fixed, s_fixed);

% Test different nEta values
nEta_values = [101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601];
etaMax = 6.0*sigmaU;
ke = 3;

results = zeros(length(nEta_values), 4);  % [nEta, Re(λ), Im(λ), |λ-1|]

for iTest = 1:length(nEta_values)
    nEta = nEta_values(iTest);
    
    % Build eta grid with stretching
    eta12 = linspace(0, (etaMax).^(1/ke), nEta);
    eta   = eta12.^ke;
    wEta  = ke * eta12.^(ke-1) .* NewtonCotes(2, eta12);
    
    % Diagnostic: show eta values near 0.01 branch threshold
    idx_near_001 = find(eta > 0.005 & eta < 0.02);
    fprintf('  eta near 0.01: ');
    fprintf('%.6f ', eta(idx_near_001));
    fprintf('\n');
    
    % Precompute orbits
    orbitData = cell(1,nEta);
    for k = 1:nEta
        orbitData{k} = compute_orbit_data(eta(k), NW_orbit, NW);
    end
    
    % Precompute Q_lmk
    Q_lmk = compute_Q_lm(alpha, eta, orbitData, m, ellVec);
    
    % Compute C_a tilde normalization
    Ca_tilde_inv = sqrt(pi) * ((a_par+1)/2)^(-a_par/2) * gamma((a_par+1)/2) * exp((a_par+1)/2) / (a_par+1);
    Ca_tilde     = 1.0 / Ca_tilde_inv;
    
    % Pack into struct
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
    
    % Compute eigenvalue
    [lambda, ~] = compute_eigenvalue(omega_fixed, precomp);
    
    results(iTest, :) = [nEta, real(lambda), imag(lambda), abs(lambda-1)];
    
    fprintf('nEta=%3d:  Re(λ)=%.12f  Im(λ)=%+.12f  |λ-1|=%.2e\n', ...
            nEta, real(lambda), imag(lambda), abs(lambda-1));
end

% Compute differences to see convergence pattern
fprintf('\n=== Differences (to see oscillation pattern) ===\n');
fprintf('nEta  |  ΔRe(λ) × 10^8  |  ΔIm(λ) × 10^8\n');
fprintf('------+----------------+----------------\n');
for iTest = 2:length(nEta_values)
    dRe = (results(iTest,2) - results(iTest-1,2)) * 1e8;
    dIm = (results(iTest,3) - results(iTest-1,3)) * 1e8;
    fprintf('%3d   |  %+12.4f  |  %+12.4f\n', nEta_values(iTest), dRe, dIm);
end

% Plot
figure('Name', 'Convergence Test');
subplot(2,1,1);
plot(nEta_values, results(:,2), 'b.-', 'MarkerSize', 15, 'LineWidth', 1.5);
xlabel('nEta'); ylabel('Re(\lambda)');
title('Real part of \lambda vs nEta');
grid on;

subplot(2,1,2);
plot(nEta_values, results(:,3), 'r.-', 'MarkerSize', 15, 'LineWidth', 1.5);
xlabel('nEta'); ylabel('Im(\lambda)');
title('Imaginary part of \lambda vs nEta');
grid on;


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
% Kalnajs gravity factor K(α,m)
% =================================================================
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
% Compute orbit data for a given eccentric velocity η
% =================================================================
function orb = compute_orbit_data(eta, NW_orbit, NW)
    
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
    for ia = 1:nAlpha
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
function F_all = compute_F_lm_all_ell(ellVec, kappa, Omega, a_par, m, omega, Hhat_h, Hhat_prime_h, exp_h, exp_matrix)
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
dh = NewtonCotes(2, h);

Hhat_h       = 1 ./ (1 + exp(-Ncut * h));
Hhat_prime_h = Ncut * exp(-Ncut * h) ./ (1 + exp(-Ncut * h)).^2;
exp_h        = exp(h);

nuVals_col = nuVals(:);
h_row      = h(:).';
exp_matrix = exp(-1i * nuVals_col * h_row) .* repmat(dh, length(nuVals_col), 1 );

Sm = zeros(nAlpha, nAlpha);

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
    
    Sm = Sm + weight_vec(kEta) * Sm_eta;
end

Sm = Kalpha.' .* Sm;
end
