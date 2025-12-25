% Matrix_LogExp_ver2.m
%
% Linear Matrix Method with alpha folded to alpha >= 0.
% We keep independent prograde/retrograde coefficients by using, for each
% positive alpha_k>0, two basis functions: W(J, alpha_k) and W^*(J, alpha_k),
% plus a single basis function at alpha=0.
%
% Discretization summary (per l):
%  - Build basis matrix X_l (NJ x Nbas):
%      col 1:                W(:, alpha=0)
%      for k=2..Npos:  col 2*(k-2)+2:  W^*(., alpha_k)
%                       col 2*(k-2)+3:  W(.,  alpha_k)
%  - Overlap/frequency blocks (Hermitian):
%      D_l = X_l^H diag(wJ) X_l
%      E_l = X_l^H diag((l*Omega1 + m*Omega2).*wJ) X_l
%  - Column quadrature weights for C: S_alpha_hat = [S_alpha(0), S_alpha(>0) repeated twice]
%  - Kernel integral over alpha (Eq. 26) with alpha>=0 weights:
%      w_alpha(k) = N(alpha_k,m) * S_alpha_pos(k) / (2*pi), (k=1..Npos)
%      F_l_beta_alpha = X_l^H diag(F0l.*wJ) conj(Wpos)     (size Nbas x Npos)
%      D_lp_alpha_alphap = Wpos^H diag(wJ) X_lp            (size Npos x Nbas')
%      B_{l,lp} = G_eff * F_l_beta_alpha * diag(w_alpha) * D_lp_alpha_alphap
%      Then multiply B columns by S_alpha_hat to apply alpha'-quadrature.
%
% Authors: E.V. Polyachenko, I.G. Shukhman (adapted)
% Date: December 2025

% clear; close all;

addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/models');
addpath('/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/Original_Matlab_C');

%% ==================== PARAMETERS ====================

m = 2;                      % azimuthal number
l_min = -7; l_max = 7;      % resonance range
l_arr = l_min:l_max; n_l = numel(l_arr);

% alpha grid on [0, alpha_max]
N_alpha_pos = 101;          % number of alpha>=0 points (include 0)
alpha_max = 10;
alpha_pos = linspace(0, alpha_max, N_alpha_pos);
S_alpha_pos = SimpsonCoef(alpha_pos);     % quadrature weights on [0, alpha_max]

% configuration-space grids
NR = 101;                   % Rc grid
Rc_ = logspace(-2, log10(20), NR);
uRc_ = log(Rc_);
S_RC = Rc_ .* SimpsonCoef(uRc_);

Ne = 31;                    % circularity grid
overboth = 0;               % retrograde toggle (0: prograde only)

% orbit integration grid
nw = 10001;                 % fine w-grid
nwa = 101;                  % reduced grid for W computation
nwai = (nw-1)/(nwa-1); Ia = 1:nwai:nw;

G = 1;                      % gravitational constant (code units)
G_eff = G * (2*pi)^2;       % normalization factor to match tests

%% ==================== MODEL ====================

fprintf('Setting up model...\n');
model = 1;  % 1: Toomre-Zang; 9: Isochrone JH
switch model
    case 1
        [V, dV, Omega, kappa, DF, DF_E, DF_L, Sigma_d, sigma_r] = GModel_toomre();
        titlestr = 'Toomre-Zang n=4, m=2';
    case 9
        [V, dV, Omega, kappa, DF, DF_E, DF_L, Sigma_d, Jr, sigma_u, Omega1E, Omega2EL] = GModel_Isochrone(12, 'JH');
        titlestr = 'Isochrone model JH m=2';
        sigma_r = sigma_u;
end
fprintf('Model: %s\n', titlestr);

%% ==================== GRIDS AND ORBITS ====================

fprintf('Setting up (Rc,v) grid...\n');
S_e = zeros(NR, Ne); v_iR = zeros(NR, Ne);
for iR = 1:NR
    if Full_ON
        v_min = -1;
    else
        v_min = 1e-2;
    end
    vv = linspace(v_min, 1, Ne);
    S_e(iR, :) = SimpsonCoef(vv);
    v_iR(iR, :) = vv;
end
v_iR(abs(v_iR)<1e-14) = 1e-12; SGNL = sign(v_iR);

e = 1 - abs(v_iR);
Rc = repmat(Rc_', 1, Ne);
R1 = Rc .* (1 - e); R2 = Rc .* (1 + e);

E = (V(R2).*R2.^2 - V(R1).*R1.^2) ./ (R2.^2 - R1.^2);
IEn = find(~isfinite(E) & e < 1); Rin = Rc(IEn); E(IEn) = Rin.*dV(Rin)/2 + V(Rin);
IEn = find(~isfinite(E)); Rin = R2(IEn); E(IEn) = V(Rin);

L2_m = 2 * (E - V(R2)) .* R2.^2; L2_m(L2_m < 1e-14) = 0;
L_m = sqrt(L2_m).*SGNL;

fprintf('Integrating orbits...');

eps_w = 1e-8; w = linspace(eps_w, pi-eps_w, nw); dw = w(2)-w(1);
cosw=cos(w); sinw=sin(w); wa=w(Ia);

Omega_1 = zeros(NR, Ne); Omega_2 = zeros(NR, Ne);
w1 = nan(nwa, NR, Ne); ra = nan(nwa, NR, Ne); pha = nan(nwa, NR, Ne);

for iR = 1:NR
  R_i = Rc(iR,1); kappa_iE=kappa(R_i); Omega_iE=Omega(R_i);
  for ie=1:Ne
    E_i=E(iR,ie); L_j=L_m(iR,ie); L2_j=L2_m(iR,ie);
    if abs(e(iR,ie)) < 1e-10
      Omega_1(iR,ie)=kappa_iE; Omega_2(iR,ie)=Omega_iE*SGNL(iR,ie);
      w1(:,iR,ie)=wa; ra(:,iR,ie)=R_i; pha(:,iR,ie)=wa*Omega_2(iR,ie)/Omega_1(iR,ie); continue
    end
    if e(iR,ie) < 0.01
      a_j=(R2(iR,ie)-R1(iR,ie))/2; Omega_1(iR,ie)=kappa_iE; Omega_2(iR,ie)=Omega_iE*SGNL(iR,ie);
      w1(:,iR,ie)=wa; ra(:,iR,ie)=R_i - a_j*cosw(Ia);
      pha(:,iR,ie)=wa*Omega_2(iR,ie)/Omega_1(iR,ie) + 2*Omega_iE/kappa_iE * a_j/R_i * sinw(Ia) * SGNL(iR,ie); continue
    end
    r1=R1(iR,ie); r2=R2(iR,ie); rs=(r1+r2)/2; drs=(r2-r1)/2; xs=rs - drs*cosw;
    if L2_j>1e-12
      rvr=sqrt(2*(E_i - V(xs)).*xs.^2 - L2_j); svr=sinw.*xs./rvr; svr(1)=svr(2)*2-svr(3); svr(end)=svr(end-1)*2-svr(end-2);
    else
      vr=sqrt(2*(E_i - V(xs))); svr=sinw./vr; svr(end)=svr(end-1)*2-svr(end-2);
    end
    dt1=drs*dw.*svr; dt2=zeros(1,nw); dt2(2:end)=(dt1(1:end-1)+dt1(2:end))/2; t=cumsum(dt2); Omega_1(iR,ie)=pi/t(end);
    w1(:,iR,ie)=t(Ia)*Omega_1(iR,ie); ra(:,iR,ie)=xs(Ia);
    if abs(1-e(iR,ie))>1e-10 && L2_j>1e-12
      svr=sinw./rvr; svr(1)=svr(2)*2-svr(3); svr(end)=svr(end-1)*2-svr(end-2);
      dt3=drs*dw.*svr./xs; dt4=zeros(1,nw); dt4(2:end)=(dt3(1:end-1)+dt3(2:end))/2; phi=cumsum(dt4); ph=L_j*phi(Ia);
    else
      ph=pi/2*ones(size(Ia))*SGNL(iR,ie); ph(1)=0;
    end
    Omega_2(iR,ie)=Omega_1(iR,ie)*ph(end)/pi; pha(:,iR,ie)=ph;
  end
  if mod(iR,10)==0, fprintf('.'); end
end
fprintf(' done.\n');

%% ==================== W_l(J, alpha>=0) ====================

fprintf('Computing W_l(J,alpha>=0)...');
W_pos = zeros(NR, Ne, N_alpha_pos, n_l);  % complex
for iR=1:NR
  for ie=1:Ne
    w1v = squeeze(w1(:,iR,ie)); rvals=squeeze(ra(:,iR,ie)); thet=squeeze(pha(:,iR,ie));
    if any(~isfinite(rvals)) || any(~isfinite(thet)) || any(~isfinite(w1v)), continue; end
    Sw1 = TrapezoidCoef(w1v')'; Om2Om1 = Omega_2(iR,ie)/Omega_1(iR,ie); phi = Om2Om1*w1v - thet;
    for il=1:n_l
      l = l_arr(il);
      for ka=1:N_alpha_pos
        a = alpha_pos(ka);
        ang = l*w1v + m*phi; logr=log(rvals);
        integrand = cos(ang) .* exp(-1i*a*logr) ./ sqrt(rvals);
        integrand(~isfinite(integrand))=0;
        W_pos(iR,ie,ka,il) = (1/pi) * sum(Sw1 .* integrand);
      end
    end
  end
  if mod(iR,10)==0, fprintf('.'); end
end
fprintf(' done.\n');

%% ==================== KERNEL N(alpha,m) (alpha>=0) ====================

ia = 1i*alpha_pos;
z1=(m+0.5+ia)/2; z2=(m+0.5-ia)/2; z3=(m+1.5+ia)/2; z4=(m+1.5-ia)/2;
Nker_pos = real(gamma_complex(z1).*gamma_complex(z2)./(gamma_complex(z3).*gamma_complex(z4)))/2;

w_alpha = (S_alpha_pos(:) .* Nker_pos(:)) / (2*pi);   % Npos x 1

%% ==================== BUILD D_l, E_l and B ====================

fprintf('Building D_l, E_l, B...\n');
NJ = NR*Ne; wJ = reshape((S_RC(:)*ones(1,Ne)).*S_e.*ones(NR,Ne).*Jacobian_placeholder(NR,Ne), [NJ,1]);
% NOTE: replace Jacobian_placeholder with your computed Jac from Matrix_LogExp.m

Omega1_vec = reshape(Omega_1,[NJ,1]); Omega2_vec = reshape(Omega_2,[NJ,1]);

% basis size per l: 1 + 2*(N_alpha_pos-1)
Nbas = 1 + 2*(N_alpha_pos-1);

% Precompute Wpos and its conjugate reshaped NJ x Npos for each l
Wpos_l = cell(n_l,1); Wpos_conj_l = cell(n_l,1);
for il=1:n_l
  Wtmp = reshape(W_pos(:,:,:,il), [NJ, N_alpha_pos]);
  Wpos_l{il} = Wtmp; Wpos_conj_l{il} = conj(Wtmp);
end

% Build per-l basis matrices X_l and blocks D_l, E_l
X_l = cell(n_l,1); D_l = cell(n_l,1); E_l = cell(n_l,1);
Falpha_l = cell(n_l,1); Dalphap_l = cell(n_l,1);
for il=1:n_l
  % assemble X_l (NJ x Nbas)
  X = zeros(NJ, Nbas);
  Wp = Wpos_l{il}; Wpc = Wpos_conj_l{il};
  X(:,1) = Wp(:,1);                            % alpha=0
  col = 2;
  for ka=2:N_alpha_pos
    X(:,col)   = Wpc(:,ka); col = col+1;       % W^*(alpha_k)
    X(:,col)   = Wp(:,ka);  col = col+1;       % W(alpha_k)
  end
  X_l{il} = X;

  % D and E blocks
  D_l{il} = X' * (wJ .* X);                    % Gram with weights
  Eloc = (l_arr(il)*Omega1_vec + m*Omega2_vec) .* wJ;
  E_l{il} = X' * (Eloc .* X);

  % F_l_beta_alpha (Nbas x Npos) and D_alpha_alphap (Npos x Nbas)
  Falpha_l{il} = X' * ((wJ .* F0l_placeholder(NJ)) .* Wpos_conj_l{il});
  Dalphap_l{il} = Wpos_l{il}' * (wJ .* X);
end

% Column quadrature weights for C: repeat positive-node weights twice
S_alpha_hat = [S_alpha_pos(1); reshape([S_alpha_pos(2:end); S_alpha_pos(2:end)], [], 1)];

% Assemble global D, E (block-diagonal) with column weights
Ntot = Nbas * n_l; D_global = zeros(Ntot, Ntot); E_global = zeros(Ntot, Ntot);
for il=1:n_l
  i1=(il-1)*Nbas+1; i2=il*Nbas;
  D_global(i1:i2,i1:i2) = D_l{il} * diag(S_alpha_hat);
  E_global(i1:i2,i1:i2) = E_l{il} * diag(S_alpha_hat);
end

% Assemble B using alpha>=0 factorization and then apply column weights diag(S_alpha_hat)
B_global = zeros(Ntot, Ntot);
for il=1:n_l
  i1=(il-1)*Nbas+1; i2=il*Nbas;
  for jl=1:n_l
    j1=(jl-1)*Nbas+1; j2=jl*Nbas;
    BlockB = Falpha_l{il} * (w_alpha(:) .* Dalphap_l{jl});  % Nbas x Nbas
    B_global(i1:i2, j1:j2) = G_eff * BlockB * diag(S_alpha_hat);
  end
end

%% ==================== EIGEN-SOLVE ====================

fprintf('Solving generalized eigenproblem...\n');
RHS = E_global + B_global;
[eigvecs, eigvals] = eig(RHS, D_global);
omega = diag(eigvals);
[~,ord] = sort(imag(omega), 'descend'); omega = omega(ord);

fprintf('\nOmega_p | gamma\n'); fprintf('----------------\n');
for i=1:min(10,numel(omega))
  fprintf('%8.5f | %10.6f\n', real(omega(i))/m, imag(omega(i)));
end

%% ==================== HELPERS ====================
function S = TrapezoidCoef(x)
    n=length(x); S=zeros(1,n); if n>=2, S(1)=(x(2)-x(1))/2; S(end)=(x(end)-x(end-1))/2; for i=2:n-1, S(i)=(x(i+1)-x(i-1))/2; end, end
end
function S = SimpsonCoef(x)
    n=length(x); if n<3, S=TrapezoidCoef(x); return; end
    dx=x(2)-x(1); S=zeros(1,n);
    if mod(n,2)==1, S(1)=dx/3; S(end)=dx/3; for i=2:2:n-1, S(i)=4*dx/3; end; for i=3:2:n-2, S(i)=2*dx/3; end
    else, S=TrapezoidCoef(x); end
end
function y = F0l_placeholder(NJ)
    % TODO: replace with vectorized (l*Omega1 + m*Omega2).*FE + m*FL of size NJ x 1
    y = ones(NJ,1);  % placeholder
end
function J = Jacobian_placeholder(NR,Ne)
    % TODO: replace with computed Jac(NR,Ne)
    J = ones(NR,Ne);
end
