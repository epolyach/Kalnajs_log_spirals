% ZANG_NEWTON_SEARCH_MP
% ---------------------------------------------------------------
% Newton-Raphson search to find (Ω_p, s) such that λ(ω) = 1
% for the m=2 mode of the Zang disk.
%
% MULTIPRECISION VERSION using Advanpix Toolbox (quadruple precision)
%
% The eigenvalue condition for a mode is det(M - I) = 0, or
% equivalently, the largest eigenvalue λ = 1.
%
% We solve F(Ω_p, s) = 0 where F = [Re(λ) - 1; Im(λ)]
% using Newton's method with finite-difference Jacobians.
% ---------------------------------------------------------------

% Set precision to quadruple (34 decimal digits)
mp.Digits(34);

% --------------- Global model parameters -----------------------
m      = mp('2');          % angular harmonic
Ncut   = mp('4');          % cut-out index N
q      = mp('6');

sigmaU = mp('1')/sqrt(q+mp('1'));   % radial velocity dispersion σ_u / V0  (Q≈1)
a_par  = q;  % 'a' in Zang's DF: a = (V0/σ_u)^2 - 1

% Wavenumber grid α (log-spiral basis)
nAlpha   = 201;       % must be odd; increase for better resolution
alphaMax = mp('10.0');     % |α| ≤ alphaMax; increase if needed
alpha    = linspace(-alphaMax, alphaMax, nAlpha);
% dAlpha   = alpha(2)-alpha(1);
dAlpha   = NewtonCotes_mp(1, alpha);

% Eccentric-velocity grid η = U/V0
nEta   = 200;         % increase for better convergence
etaMax = mp('5.0')*sigmaU;  % covers most of the Maxwellian

eta    = linspace(mp('0'), etaMax, nEta);
wEta   = NewtonCotes(2, eta);

% Radial Fourier harmonics ℓ range
ellMin = -25;
ellMax =  25;
ellVec = ellMin:ellMax;
nEll   = numel(ellVec);

% Number of orbital phase samples per radial period
NW_orbit = 100001;
NW       = 201;

% Build Kalnajs gravity factor K(α,m)
Kalpha  = kalnajs_K(alpha, m);


% ---------- Precompute orbits and Fourier coefficients ----------
fprintf('Precomputing orbits and Q_{ℓm}(α;η)...\n');
orbitData = cell(1,nEta);
for k = 1:nEta
    orbitData{k} = compute_orbit_data(eta(k), NW_orbit, NW);
end

Q_lmk = compute_Q_lm(alpha, eta, orbitData, m, ellVec);

% ---------- Compute C_a tilde normalization (analytic, eq. 2.63) ----------
Ca_tilde_inv = sqrt(mp('pi')) * ((a_par+mp('1'))/mp('2'))^(-a_par/mp('2')) * gamma((a_par+mp('1'))/mp('2')) * exp((a_par+mp('1'))/mp('2')) / (a_par+mp('1'));
Ca_tilde     = mp('1') / Ca_tilde_inv ;

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
Omega_p = mp('0.439426');
s       = mp('0.127181');

% Newton parameters
max_iter = 50;
tol = mp('1e-16');  % tighter tolerance for quad precision
delta = mp('1e-8');  % smaller finite difference step for better accuracy

fprintf('Initial: Ω_p = %.16f, s = %.16f\n', double(Omega_p), double(s));
fprintf('Target:  λ = 1 + 0i\n\n');

convergence_history = zeros(max_iter, 5, 'mp');  % [Omega_p, s, Re(λ), Im(λ), |F|]

for iter = 1:max_iter
    omega = m*Omega_p + mp('1i')*s;
    
    % Compute λ at current point
    [lambda, vec] = compute_eigenvalue(omega, precomp);
    
    % Residual: want λ = 1, so F = [Re(λ)-1; Im(λ)]
    F = [real(lambda) - mp('1'); imag(lambda)];
    res_norm = norm(F);
    
    convergence_history(iter,:) = [Omega_p, s, real(lambda), imag(lambda), res_norm];
    
    fprintf('Iter %2d: Ω_p=%.16f  s=%.16f  λ=%.16f%+.16fi  |F|=%.2e\n', ...
            iter, double(Omega_p), double(s), double(real(lambda)), double(imag(lambda)), double(res_norm));
    
    % Check convergence
    if res_norm < tol
        fprintf('\n*** Converged! ***\n');
        convergence_history = convergence_history(1:iter,:);
        break;
    end
    
    % Compute Jacobian via finite differences
    % J = [dRe(λ)/dΩ_p, dRe(λ)/ds; dIm(λ)/dΩ_p, dIm(λ)/ds]
    
    % Partial w.r.t. Ω_p
    omega_dOp = m*(Omega_p + delta) + mp('1i')*s;
    lambda_dOp = compute_eigenvalue(omega_dOp, precomp);
    dF_dOp = ([real(lambda_dOp); imag(lambda_dOp)] - [real(lambda); imag(lambda)]) / delta;
    
    % Partial w.r.t. s
    omega_ds = m*Omega_p + mp('1i')*(s + delta);
    lambda_ds = compute_eigenvalue(omega_ds, precomp);
    dF_ds = ([real(lambda_ds); imag(lambda_ds)] - [real(lambda); imag(lambda)]) / delta;
    
    J = [dF_dOp, dF_ds];
    
    % Check Jacobian condition
    cond_J = cond(J);
    if cond_J > mp('1e10')
        warning('Jacobian ill-conditioned (cond = %.2e), reducing step', double(cond_J));
    end
    
    % Newton step: x_new = x - J \ F
    dx = J \ F;
    
    % Line search (backtracking if needed)
    step = mp('1.0');
    for ls = 1:10
        Omega_p_new = Omega_p - step * dx(1);
        s_new = s - step * dx(2);
        
        % Ensure s > 0 (growth rate must be positive for unstable modes)
        if s_new > mp('1e-4') && Omega_p_new > mp('0') && Omega_p_new < mp('1')
            omega_new = m*Omega_p_new + mp('1i')*s_new;
            lambda_new = compute_eigenvalue(omega_new, precomp);
            F_new = [real(lambda_new) - mp('1'); imag(lambda_new)];
            if norm(F_new) < res_norm || ls == 10
                break;
            end
        end
        step = step * mp('0.5');
    end
    
    if step < mp('1')
        fprintf('         (backtracking: step = %.3f)\n', double(step));
    end
    
    Omega_p = Omega_p - step * dx(1);
    s = s - step * dx(2);
end

% Final result
omega = m*Omega_p + mp('1i')*s;
[lambda_final, vec] = compute_eigenvalue(omega, precomp);

fprintf('\n=== Final Result ===\n');
fprintf('Ω_p = %.16f   (Zang: 0.439426)\n', double(Omega_p));
fprintf('s   = %.16f   (Zang: 0.127181)\n', double(s));
fprintf('λ   = %.16f + %.16fi\n', double(real(lambda_final)), double(imag(lambda_final)));
fprintf('|λ-1| = %.2e\n', double(abs(lambda_final - mp('1'))));

% Plot convergence
% figure('Name', 'Newton Convergence');
% subplot(2,1,1);
% semilogy(1:size(convergence_history,1), double(convergence_history(:,5)), 'b.-', 'LineWidth', 1.5, 'MarkerSize', 15);
% xlabel('Iteration');
% ylabel('|F| = |[\Re(\lambda)-1, \Im(\lambda)]|');
% title('Newton Convergence');
% grid on;
% 
% subplot(2,1,2);
% plot(double(convergence_history(:,1)), double(convergence_history(:,2)), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
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
alpha = alpha(:).';       % row vector
ia    = mp('1i') * alpha;

z1 = (m + mp('0.5') + ia)/mp('2');
z2 = (m + mp('0.5') - ia)/mp('2');
z3 = (m + mp('1.5') + ia)/mp('2');
z4 = (m + mp('1.5') - ia)/mp('2');

K = gamma(z1).*gamma(z2) ./ (gamma(z3).*gamma(z4));
K = real(K)/mp('2');              % should be purely real & positive
end

% =================================================================
% Compute orbit data for a given eccentric velocity η
% =================================================================
function orb = compute_orbit_data(eta, NW_orbit, NW);
    
Omega_bar = mp('1');
kappa_bar = sqrt(mp('2.0'));
I0 = mp('2')*mp('pi')/kappa_bar;
I1 = I0;
I2 = I0;

w = linspace(mp('0'), mp('pi'), NW_orbit);
d_w = w(2) - w(1);
cos_w = cos(w);
sin_w = sin(w);

if eta < mp('1e-10')
    phi = w * Omega_bar/kappa_bar;
    xs = mp('1') - mp('0')*cos_w;
    t = w/kappa_bar;
    X = log(xs);
    Y = phi - Omega_bar * t;
    psi = w;

elseif eta < mp('0.01')
    a = eta/sqrt(mp('2'));
    xs = mp('1') - a*cos_w;
    phi = w*Omega_bar/kappa_bar + mp('2')*Omega_bar/kappa_bar*a*sin_w;
    t = w/kappa_bar;
    X = log(xs);
    Y = phi - Omega_bar * t;
    psi = w;

else
    [rmin, rmax] = turning_points(eta);
    rs = (rmax+rmin)/mp('2');
    ds = (rmax-rmin)/mp('2');

    xs = rs - ds*cos_w;
    rvr = sqrt(xs.^2 .* (eta.^2 + mp('1') - mp('2')*log(xs)) - mp('1'));

    svr = sin_w.*xs./rvr;
    svr(1)   = svr(2)*mp('2')     - svr(3);
    svr(end) = svr(end-1)*mp('2') - svr(end-2);

    dt1 = ds * d_w .* svr;
    dt2 = zeros(1, NW_orbit, 'mp');
    dt2(2:end) = (dt1(1:end-1)+dt1(2:end))/mp('2');
    t_half = cumsum(dt2);
    T_half = t_half(end);
    kappa_bar = mp('pi')/T_half;
    X = log(xs);
    psi = kappa_bar*t_half;

    dphi1 = dt1./xs.^2;
    dphi2 = zeros(1, NW_orbit, 'mp');
    dphi2(2:end) = (dphi1(1:end-1)+dphi1(2:end))/mp('2');
    phi_half = cumsum(dphi2);
    Omega_bar = phi_half(end)/T_half;
    Y = phi_half - Omega_bar * t_half;

    I0 = mp('2')*T_half;
    I1 = mp('2')*sum(dt1./xs);
    I2 = mp('2')*phi_half(end);

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
f = @(r) eta.^2 + mp('1') - mp('2')*log(r) - r.^(-2);
rmin = fzero(f, [mp('0.1'), mp('1.0')]);
rmax = fzero(f, [mp('1.0'), mp('10.0')]);
end


% =================================================================
% Compute Q_{ℓm}(α;η) - VECTORIZED
% =================================================================
function Q = compute_Q_lm(alpha, eta, orbitData, m, ellVec)
nAlpha = numel(alpha);
nEta   = numel(eta);
nEll   = numel(ellVec);
Q = zeros(nEll, nAlpha, nEta, 'mp');

ell_col   = ellVec(:);

for kEta = 1:nEta
    orb  = orbitData{kEta};
    X    = orb.X(:).';
    Y    = orb.Y(:).';
    psi  = orb.psi(:).';
    dpsi = TrapNE_Coef(psi);
    for ia = 1:nAlpha,
        alpha_i = alpha(ia);
        base_kernel = exp((mp('1i') * alpha_i-mp('0.5'))*X ) .* dpsi;
        ell_phase = cos(ell_col * psi - m*repmat(Y,length(ell_col),1));
        Q(:,ia,kEta) = (ell_phase * base_kernel.')/mp('pi');
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
numerator = ((a_par + mp('1')) * freq_term - a_par * m) .* Hhat_h - m .* Hhat_prime_h;
denominator = freq_term - omega .* exp_h;
integrand_base = numerator ./ denominator;

F_all = (integrand_base * exp_matrix.') / (mp('2')*mp('pi'));
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

I0_vec    = zeros(1, nEta, 'mp');
kappa_vec = zeros(1, nEta, 'mp');
Omega_vec = zeros(1, nEta, 'mp');
for kEta = 1:nEta
    orb = orbitData{kEta};
    I0_vec(kEta)    = orb.I0;
    kappa_vec(kEta) = orb.kappa;
    Omega_vec(kEta) = orb.Omega;
end
gauss_eta_vec = exp(-eta.^2 / (mp('2')*sigmaU^2));
weight_vec    = Ca_tilde * I0_vec .* wEta .* eta .* gauss_eta_vec;

% Precompute h-grid quantities
hmax = mp('4.0');
Nh   = 8192;
h    = linspace(-hmax, hmax, Nh);
% dh   = h(2) - h(1);
dh = NewtonCotes(2, h);

Hhat_h       = mp('1') ./ (mp('1') + exp(-Ncut * h));
Hhat_prime_h = Ncut * exp(-Ncut * h) ./ (mp('1') + exp(-Ncut * h)).^2;
exp_h        = exp(h);

nuVals_col = nuVals(:);
h_row      = h(:).';
exp_matrix = exp(-mp('1i') * nuVals_col * h_row) .* repmat(dh, length(nuVals_col), 1 );

Sm = zeros(nAlpha, nAlpha, 'mp');

for kEta = 1:nEta
    F_all = compute_F_lm_all_ell(ellVec, kappa_vec(kEta), Omega_vec(kEta), ...
                                 a_par, m, omega, Hhat_h, Hhat_prime_h, exp_h, exp_matrix);
    
    Sm_eta = zeros(nAlpha, nAlpha, 'mp');
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
