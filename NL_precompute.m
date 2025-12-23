% NL_PRECOMPUTE
%
% Precompute all ω-independent quantities for the nonlinear eigenvalue
% method (Eqs. 39-46).
%
% Output (to workspace):
%   data - struct containing all precomputed quantities needed for
%          building M(β,α;ω) and finding zeros of det(I - M·dα)
%
% After running this, run NL_grid_scan and/or NL_newton_search

addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/models');
addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/Original_Matlab_C');

%% ==================== PARAMETERS ====================
% Modify these as needed

m         = 2;       % Azimuthal number
l_min     = -25;     % Min radial harmonic
l_max     = 25;      % Max radial harmonic
G         = 1;       % Gravitational constant
N_alpha   = 101;     % Alpha grid points
alpha_max = 10;      % Max alpha
NR        = 201;     % Radial grid points
Ne        = 21;      % Eccentricity grid points
Full_ON   = 0;       % 0: prograde only
model     = 1;       % 1: Toomre-Zang
nw        = 10001;  % Orbit integration points
nwa       = 201;     % Reduced orbit grid

n_l = l_max - l_min + 1;
l_arr = l_min:l_max;

% Alpha grid
alpha_arr = linspace(-alpha_max, alpha_max, N_alpha);
d_alpha   = TrapezoidCoef(alpha_arr); %alpha_arr(2) - alpha_arr(1);
alpha_integ = 't';  % 't' for trapezoidal

% Rc grid (logarithmic)
Rc_ = logspace(-3, 2.0, NR);
Rc_min = min(Rc_);
Rc_max = max(Rc_);
uRc_ = log(Rc_);
S_RC = Rc_ .* TrapezoidCoef(uRc_); % SimpsonCoef(uRc_);
Rc_integ = 't';  % 't' for trapezoidal, 's' for Simpson

nwai = (nw-1)/(nwa-1);
Ia = 1:nwai:nw;
NPh = NR*Ne;

%% ==================== MODEL SETUP ====================

fprintf('Setting up model...\n');

switch model
    case 1
        [V, dV, Omega, kappa, DF, DF_E, DF_L, ~, ~] = GModel_toomre();
        titlestr = 'Toomre-Zang n=4, m=2';
end

fprintf('Model: %s\n', titlestr);

%% ==================== GRID SETUP ====================

fprintf('Setting up grids...\n');

S_e = zeros(NR, Ne);
v_iR = zeros(NR, Ne);

% Set v_min (stored for reporting)
if Full_ON
    v_min = -1;
else
    v_min = 1e-2;
end
v_integ = 's';  % 's' for Simpson

for iR = 1:NR
    v = linspace(v_min, 1, Ne);
    S_e(iR, :) = SimpsonCoef(v);
    v_iR(iR, :) = v;
end

v_iR(abs(v_iR) < 1e-14) = 1e-12;
SGNL = sign(v_iR);

e = 1 - abs(v_iR);
Rc = repmat(Rc_', 1, Ne);
R1 = Rc .* (1 - e);
R2 = Rc .* (1 + e);

E = (V(R2) .* R2.^2 - V(R1) .* R1.^2) ./ (R2.^2 - R1.^2);

IEn = find(~isfinite(E) & e < 1);
Rin = Rc(IEn);
E(IEn) = Rin .* dV(Rin) / 2 + V(Rin);

IEn = find(~isfinite(E));
Rin = R2(IEn);
E(IEn) = V(Rin);

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
        E_i  = E(iR, ie);
        L_j  = L_m(iR, ie);
        L2_j = L2_m(iR, ie);
        
        if abs(e(iR, ie)) < 1e-10
            Omega_1(iR, ie) = kappa_iE;
            Omega_2(iR, ie) = Omega_iE * SGNL(iR, ie);
            w1(:, iR, ie) = wa;
            ra(:, iR, ie) = R_i;
            pha(:, iR, ie) = wa * Omega_2(iR, ie) / Omega_1(iR, ie);
            continue;
        end
        
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
    
    if mod(iR, 20) == 0
        fprintf('.');
    end
end
fprintf(' done.\n');

%% ==================== JACOBIAN CALCULATION ====================

fprintf('Computing Jacobian...\n');

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
        
        general_orbit = (abs(r2 - r1) > 1e-12) && (abs(L_val) > 1e-12);
        if general_orbit
            V1 = V(r1);
            V2 = V(r2);
            dV1 = dV(r1);
            dV2 = dV(r2);
            
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

F0(~isfinite(F0)) = 0;
FE(~isfinite(FE)) = 0;
FL(~isfinite(FL)) = 0;

%% ==================== COMPUTE W_l BASIS FUNCTIONS (Eq. 41) ====================

fprintf('Computing W_l basis functions (Eq. 41)...\n');

W_l = zeros(NR, Ne, N_alpha, n_l);

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
    
    if mod(iR, 20) == 0
        fprintf('.');
    end
end
fprintf(' done.\n');

%% ==================== COMPUTE N_m(α) KERNEL (Eq. 46) ====================

fprintf('Computing N_m(α) kernel (Eq. 46)...\n');

ia = 1i * alpha_arr;

z1 = (m + 0.5 + ia) / 2;
z2 = (m + 0.5 - ia) / 2;
z3 = (m + 1.5 + ia) / 2;
z4 = (m + 1.5 - ia) / 2;

N_kernel = real(gamma_complex(z1) .* gamma_complex(z2) ./ ...
               (gamma_complex(z3) .* gamma_complex(z4))) * pi .* d_alpha;

%% ==================== PRECOMPUTE ω-INDEPENDENT QUANTITIES ====================

fprintf('Precomputing ω-independent quantities...\n');

% Integration weights: dJ = Jacobian * S_RC * S_e
DJ = repmat(S_RC.', 1, Ne) .* S_e .* Jac;
DJ_vec = reshape(DJ, NPh, 1);

% F_{0,l}(J) = (l*Ω_1 + m*Ω_2) * ∂F0/∂E + m * ∂F0/∂L
F0l_all = zeros(NPh, n_l);
for i_l = 1:n_l
    l = l_arr(i_l);
    F0l = (l * Omega_1 + m * Omega_2) .* FE + m * FL;
    F0l_all(:, i_l) = reshape(F0l, NPh, 1);
end

% Resonance frequencies: l*Ω_1 + m*Ω_2
Omega_res = zeros(NPh, n_l);
for i_l = 1:n_l
    l = l_arr(i_l);
    Omega_res(:, i_l) = reshape(l * Omega_1 + m * Omega_2, NPh, 1);
end

% W_l matrices reshaped for efficiency
W_l_mat = zeros(NPh, N_alpha, n_l);
for i_l = 1:n_l
    W_l_mat(:, :, i_l) = reshape(W_l(:, :, :, i_l), NPh, N_alpha);
end

fprintf('Precomputation complete.\n');

%% ==================== PACK OUTPUT STRUCT ====================

data.m         = m;
data.G         = G;
data.N_alpha   = N_alpha;
data.alpha_max = alpha_max;
data.l_min     = l_min;
data.l_max     = l_max;
data.n_l       = n_l;
data.NR        = NR;
data.Ne        = Ne;
data.v_min     = v_min;
data.Rc_min    = Rc_min;
data.Rc_max    = Rc_max;
data.NPh       = NPh;
data.nw        = nw;
data.nwa       = nwa;
data.integ     = [alpha_integ Rc_integ v_integ];
data.alpha_arr = alpha_arr;
data.N_kernel  = N_kernel;
data.DJ_vec    = DJ_vec;
data.F0l_all   = F0l_all;
data.Omega_res = Omega_res;
data.W_l_mat   = W_l_mat;
data.titlestr  = titlestr;

fprintf('\n=== Precomputation done. ''data'' struct is in workspace. ===\n');
fprintf('Next: run NL_grid_scan or NL_newton_search\n');

%% ==================== LOCAL HELPER FUNCTIONS ====================

function S = TrapezoidCoef(x)
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
    n = length(x);
    if n < 3
        S = TrapezoidCoef(x);
        return;
    end
    
    dx = x(2) - x(1);
    S = zeros(1, n);
    
    if mod(n, 2) == 1
        S(1) = dx/3;
        S(end) = dx/3;
        for i = 2:2:n-1
            S(i) = 4*dx/3;
        end
        for i = 3:2:n-2
            S(i) = 2*dx/3;
        end
    else
        S = TrapezoidCoef(x);
    end
end
