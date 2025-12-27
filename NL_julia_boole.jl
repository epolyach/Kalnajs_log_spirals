#!/usr/bin/env julia

const HELP_TEXT = """
Kalnajs Log-Spiral CPU Solver - Boole Support

Usage: julia --project=. NL_julia_boole.jl config.toml model.toml [options]

Arguments:
  config.toml    Configuration file path
  model.toml     Model file path

Options:
  --threads=N    BLAS threads limit (default: from config)
  --quad=STR     Quadrature for alpha-Rc-v (default: "tts")
                 Chars: t=Trapezoid, s=Simpson, b=Boole
                 Example: --quad=ttb (Trapezoid for alpha/Rc, Boole for v)

Note on Boole's Rule: Requires number of points N such that (N-1) is divisible by 4.
If not divisible, it falls back to Simpson.
"""

start_time = time()

using LinearAlgebra
using SpecialFunctions
using Printf
using TOML

# ==================== CLI PARSING ====================
function parse_args()
    if length(ARGS) < 2
        println("Error: Insufficient arguments")
        println(HELP_TEXT)
        exit(1)
    end
    
    if ARGS[1] == "--help" || ARGS[1] == "-h"
        println(HELP_TEXT)
        exit(0)
    end
    
    config_file = ARGS[1]
    model_file = ARGS[2]
    threads_override = nothing
    quad_str = "tts" # Default
    
    for arg in ARGS[3:end]
        if startswith(arg, "--threads=")
            threads_override = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--quad=")
            quad_str = split(arg, "=")[2]
        elseif arg == "--help" || arg == "-h"
            println(HELP_TEXT)
            exit(0)
        end
    end
    
    if length(quad_str) != 3
        println("Error: --quad must be 3 characters (e.g. 'tts')")
        exit(1)
    end

    return config_file, model_file, threads_override, quad_str
end

config_file, model_file, threads_override, quad_str = parse_args()

# Validate input files
if !isfile(config_file)
    println("Error: Configuration file not found: $config_file")
    exit(1)
end
if !isfile(model_file)
    println("Error: Model file not found: $model_file")
    exit(1)
end

# ==================== LOAD CONFIG ====================
println("Loading config: $config_file")
println("Loading model:  $model_file")
config = TOML.parsefile(config_file)
model_data = TOML.parsefile(model_file)

# Apply thread limit
max_threads = get(get(config, "cpu", Dict()), "max_threads", 4)
if threads_override !== nothing
    max_threads = min(threads_override, max_threads)
end
BLAS.set_num_threads(max_threads)

# ==================== PRECISION ====================
const USE_FLOAT32 = !get(config["precision"], "cpu_double_precision", true)
const FT = USE_FLOAT32 ? Float32 : Float64
const CT = USE_FLOAT32 ? ComplexF32 : ComplexF64

# ==================== SETUP ====================
println("=" ^ 60)
println("KALNAJS LOG-SPIRAL CPU SOLVER (Boole Enabled)")
println("=" ^ 60)
println("Config: $config_file")
println("Model:  $model_file")
println("Quad:   $quad_str (alpha-Rc-v)")
println("Precision: ", USE_FLOAT32 ? "Float32" : "Float64")
println("BLAS threads: $max_threads")
println()

# ==================== PARAMETERS FROM CONFIG AND MODEL ====================
# From model file
const m = model_data["physics"]["m"]
const G = FT(model_data["model"]["G"])
const n_zang = model_data["model"]["n_zang"]
const q1 = model_data["model"]["q1"]
const ref_Omega_p = get(get(model_data, "reference", Dict()), "Omega_p", 0.44)
const ref_gamma = get(get(model_data, "reference", Dict()), "gamma", 0.13)

# From config file
const l_min = config["grid"]["l_min"]
const l_max = config["grid"]["l_max"]
const N_alpha = config["grid"]["N_alpha"]
const alpha_max = FT(config["grid"]["alpha_max"])
const NR = config["grid"]["NR"]
const Ne = config["grid"]["Ne"]
const Rc_min = config["grid"]["Rc_min"]
const Rc_max = config["grid"]["Rc_max"]
const v_min_cfg = config["grid"]["v_min"]
const nw = config["grid"]["nw"]
const nwa = config["grid"]["nwa"]

const n_l = l_max - l_min + 1
const l_arr = collect(l_min:l_max)
const NPh = NR * Ne

const max_iter = get(config["newton"], "max_iter", 30)
const newton_tol = USE_FLOAT32 ? 1e-7 : get(config["newton"], "tol", 1e-10)
const delta = get(config["newton"], "delta", 1e-6)

println("Model: n_zang=$n_zang, q1=$q1")
println("Grid: NR=$NR, Ne=$Ne, N_alpha=$N_alpha, l=[$l_min,$l_max], NPh=$NPh")

# ==================== TOOMRE-ZANG MODEL ====================
const L0 = 1.0
const sigma_r0 = 1.0 / sqrt(q1)
const q_zang = q1 - 1
const CONST_DF = 1.0 / (2.0 * π * 2.0^(q_zang/2) * sqrt(π) * gamma((q_zang + 1)/2) * sigma_r0^(q_zang + 2))

V(r) = log(r)
dV(r) = 1.0 / r
Omega_func(r) = 1.0 / r
kappa_func(r) = sqrt(2.0) / r
taper_func(L) = n_zang == 0 ? 1.0 : (abs(L) < 1e-12 ? 0.0 : 1.0 / (1.0 + (L0 / abs(L))^n_zang))

function DF(E, L)
    L_abs = abs(L)
    L_abs < 1e-12 && return 0.0
    result = CONST_DF * exp(-E / sigma_r0^2) * L_abs^q_zang * taper_func(L_abs)
    return isfinite(result) ? result : 0.0
end
DF_E(E, L) = -DF(E, L) / sigma_r0^2
function DF_L(E, L)
    L_abs = abs(L)
    L_abs < 1e-12 && return 0.0
    df_val = DF(E, L)
    taper_val = taper_func(L_abs)
    return df_val * (q_zang + n_zang - n_zang * taper_val) / L_abs
end

# ==================== HELPER FUNCTIONS ====================
function TrapezoidCoef(x)
    n = length(x); S = zeros(n)
    n >= 2 || return S
    S[1] = (x[2] - x[1]) / 2; S[end] = (x[end] - x[end-1]) / 2
    for i in 2:n-1; S[i] = (x[i+1] - x[i-1]) / 2; end
    return S
end

function SimpsonCoef(x)
    n = length(x)
    (n < 3 || mod(n, 2) == 0) && return TrapezoidCoef(x)
    dx = x[2] - x[1]; S = zeros(n)
    S[1] = dx / 3; S[end] = dx / 3
    for i in 2:2:n-1; S[i] = 4 * dx / 3; end
    for i in 3:2:n-2; S[i] = 2 * dx / 3; end
    return S
end

function BooleCoef(x)
    n = length(x)
    # Requires N = 4k + 1. If not, fallback to Simpson
    if (n - 1) % 4 != 0
        return SimpsonCoef(x)
    end
    dx = x[2] - x[1]
    S = zeros(n)
    # Boole's Rule block (5 points): 2h/45 * (7, 32, 12, 32, 7)
    factor = 2 * dx / 45
    # Overlapping points sum up (7+7=14)
    
    # Start/End
    S[1] = 7 * factor
    S[end] = 7 * factor
    
    # Loop over blocks
    # Pattern repeats every 4 steps
    # Internal points 
    # i corresponds to index in 1-based array
    # pattern: 7, 32, 12, 32, 14, 32, 12, 32, 14 ...
    
    for i in 2:n-1
        rem_idx = (i - 1) % 4
        if rem_idx == 0 # Multiples of 4 (indices 5, 9...) -> Shared node (7+7=14)
            S[i] = 14 * factor
        elseif rem_idx == 1 || rem_idx == 3 # indices 2, 4, 6, 8... -> 32
            S[i] = 32 * factor
        elseif rem_idx == 2 # indices 3, 7... -> 12
            S[i] = 12 * factor
        end
    end
    return S
end

function GetCoefs(x, method_char)
    if method_char == 't'
        return TrapezoidCoef(x)
    elseif method_char == 's'
        return SimpsonCoef(x)
    elseif method_char == 'b'
        return BooleCoef(x)
    else
        println("Warning: Unknown quadrature '$method_char', defaulting to Trapezoid")
        return TrapezoidCoef(x)
    end
end

# ==================== CPU PRECOMPUTATION ====================
println("Setting up grids...")
alpha_arr = collect(range(-alpha_max, alpha_max, length=N_alpha))
d_alpha = GetCoefs(alpha_arr, quad_str[1]) # Alpha quad

Rc_ = 10.0 .^ range(log10(Rc_min), log10(Rc_max), length=NR)
# Log-space integration for Rc means we integrate f(Rc) dRc = f(Rc) * Rc * d(ln Rc)
# So we apply coefs to log(Rc) grid, then multiply by Rc
S_RC = Rc_ .* GetCoefs(log.(Rc_), quad_str[2]) # Rc quad

nwai = (nw - 1) ÷ (nwa - 1); Ia = 1:nwai:nw

S_e = zeros(NR, Ne); v_iR = zeros(NR, Ne)
for iR in 1:NR
    v = collect(range(v_min_cfg, 1.0, length=Ne))
    S_e[iR, :] .= GetCoefs(v, quad_str[3]) # v quad
    v_iR[iR, :] .= v
end
v_iR[abs.(v_iR) .< 1e-14] .= 1e-12
SGNL = sign.(v_iR); e = 1.0 .- abs.(v_iR)
Rc = repeat(reshape(Rc_, :, 1), 1, Ne)
R1 = Rc .* (1.0 .- e); R2 = Rc .* (1.0 .+ e)

E_grid = (V.(R2) .* R2.^2 .- V.(R1) .* R1.^2) ./ (R2.^2 .- R1.^2)
for idx in findall(.!isfinite.(E_grid) .& (e .< 1)); E_grid[idx] = Rc[idx] * dV(Rc[idx]) / 2 + V(Rc[idx]); end
for idx in findall(.!isfinite.(E_grid)); E_grid[idx] = V(R2[idx]); end
L2_m = 2.0 .* (E_grid .- V.(R2)) .* R2.^2; L2_m[L2_m .< 1e-14] .= 0.0
L_m = sqrt.(L2_m) .* SGNL

println("Computing orbits...")
eps_w = 1e-8; w = collect(range(eps_w, π - eps_w, length=nw))
dw = w[2] - w[1]; cosw, sinw, wa = cos.(w), sin.(w), w[Ia]
Omega_1 = zeros(NR, Ne); Omega_2 = zeros(NR, Ne)
w1 = fill(NaN, nwa, NR, Ne); ra = fill(NaN, nwa, NR, Ne); pha = fill(NaN, nwa, NR, Ne)

for iR in 1:NR
    R_i = Rc[iR, 1]; kappa_iE, Omega_iE = kappa_func(R_i), Omega_func(R_i)
    for ie in 1:Ne
        E_i, L_j, L2_j = E_grid[iR, ie], L_m[iR, ie], L2_m[iR, ie]
        if abs(e[iR, ie]) < 1e-10
            Omega_1[iR, ie] = kappa_iE; Omega_2[iR, ie] = Omega_iE * SGNL[iR, ie]
            w1[:, iR, ie] .= wa; ra[:, iR, ie] .= R_i
            pha[:, iR, ie] .= wa .* Omega_2[iR, ie] / Omega_1[iR, ie]; continue
        end
        if e[iR, ie] < 0.01
            a_j = (R2[iR, ie] - R1[iR, ie]) / 2
            Omega_1[iR, ie] = kappa_iE; Omega_2[iR, ie] = Omega_iE * SGNL[iR, ie]
            w1[:, iR, ie] .= wa; ra[:, iR, ie] .= R_i .- a_j .* cosw[Ia]
            pha[:, iR, ie] .= wa .* Omega_2[iR, ie] / Omega_1[iR, ie] .+ 2 * Omega_iE / kappa_iE * a_j / R_i .* sinw[Ia] .* SGNL[iR, ie]; continue
        end
        r1, r2 = R1[iR, ie], R2[iR, ie]; rs, drs = (r1 + r2) / 2, (r2 - r1) / 2
        xs = rs .- drs .* cosw
        local rvr, svr
        if L2_j > 1e-12
            rvr = sqrt.(max.(0.0, 2.0 .* (E_i .- V.(xs)) .* xs.^2 .- L2_j))
            svr = sinw .* xs ./ rvr; svr[1] = 2 * svr[2] - svr[3]; svr[end] = 2 * svr[end-1] - svr[end-2]
        else
            vr = sqrt.(max.(0.0, 2.0 .* (E_i .- V.(xs)))); svr = sinw ./ vr; svr[end] = 2 * svr[end-1] - svr[end-2]
        end
        dt1 = drs .* dw .* svr; dt2 = zeros(nw); dt2[2:end] .= (dt1[1:end-1] .+ dt1[2:end]) ./ 2
        t = cumsum(dt2); Omega_1[iR, ie] = π / t[end]
        w1[:, iR, ie] .= t[Ia] .* Omega_1[iR, ie]; ra[:, iR, ie] .= xs[Ia]
        local ph
        if abs(1 - e[iR, ie]) > 1e-10 && L2_j > 1e-12
            svr_phi = sinw ./ rvr; svr_phi[1] = 2 * svr_phi[2] - svr_phi[3]; svr_phi[end] = 2 * svr_phi[end-1] - svr_phi[end-2]
            dt3 = drs .* dw .* svr_phi ./ xs; dt4 = zeros(nw); dt4[2:end] .= (dt3[1:end-1] .+ dt3[2:end]) ./ 2
            ph = L_j .* cumsum(dt4)[Ia]
        else; ph = π/2 .* ones(length(Ia)) .* SGNL[iR, ie]; ph[1] = 0.0; end
        Omega_2[iR, ie] = Omega_1[iR, ie] * ph[end] / π; pha[:, iR, ie] .= ph
    end
    mod(iR, 50) == 0 && print(".")
end
println(" done.")

println("Computing Jacobian & DF...")
Jac = zeros(NR, Ne)
for iR in 1:NR, ie in 1:Ne
    r1, r2, rc = R1[iR, ie], R2[iR, ie], Rc[iR, ie]
    E_val, L_val, omega1 = E_grid[iR, ie], L_m[iR, ie], Omega_1[iR, ie]
    if (abs(r2 - r1) > 1e-12) && (abs(L_val) > 1e-12)
        V1, V2, dV1, dV2 = V(r1), V(r2), dV(r1), dV(r2)
        t1 = 2 * (E_val - V1) * r1 - dV1 * r1^2; t2 = 2 * (E_val - V2) * r2 - dV2 * r2^2
        Jac[iR, ie] = 2 * abs(t1 * t2 / (r2^2 - r1^2)) * rc / omega1 / abs(L_val)
    elseif abs(L_val) <= 1e-12
        DelE = 2.0 * (E_val - V(r1)); Jac[iR, ie] = sqrt(max(DelE, 0.0)) * dV(r2) * 2 * rc / omega1
        !isfinite(Jac[iR, ie]) && (Jac[iR, ie] = 0.0)
    end
end

F0, FE, FL = zeros(NR, Ne), zeros(NR, Ne), zeros(NR, Ne)
for iR in 1:NR, ie in 1:Ne
    F0[iR, ie] = DF(E_grid[iR, ie], L_m[iR, ie]); FE[iR, ie] = DF_E(E_grid[iR, ie], L_m[iR, ie]); FL[iR, ie] = DF_L(E_grid[iR, ie], L_m[iR, ie])
end
F0[.!isfinite.(F0)] .= 0.0; FE[.!isfinite.(FE)] .= 0.0; FL[.!isfinite.(FL)] .= 0.0

println("Computing W_l basis functions ($(Threads.nthreads()) threads)...")
W_l = zeros(CT, NR, Ne, N_alpha, n_l)
# Parallelize over phase-space points
indices = [(iR, ie) for iR in 1:NR for ie in 1:Ne]
Threads.@threads for idx in 1:length(indices)
    iR, ie = indices[idx]
    w1_v = w1[:, iR, ie]; ra_v = ra[:, iR, ie]; pha_v = pha[:, iR, ie]
    if !all(isfinite.(w1_v)) || !all(isfinite.(ra_v)); continue; end
    Sw1 = TrapezoidCoef(w1_v)
    Om2_Om1 = Omega_2[iR, ie] / Omega_1[iR, ie]
    phi_vals = Om2_Om1 .* w1_v .- pha_v
    for i_l in 1:n_l, i_alpha in 1:N_alpha
        l, alpha = l_arr[i_l], alpha_arr[i_alpha]
        integrand = cos.(l .* w1_v .+ m .* phi_vals) .* exp.(-im .* alpha .* log.(ra_v)) ./ sqrt.(ra_v)
        W_l[iR, ie, i_alpha, i_l] = sum(Sw1 .* integrand) / π
    end
end
W_l_mat = reshape(W_l, NPh, N_alpha, n_l)

ia = im .* alpha_arr
N_kernel_f64 = real.(gamma.((m + 0.5 .+ ia) ./ 2) .* gamma.((m + 0.5 .- ia) ./ 2) ./ (gamma.((m + 1.5 .+ ia) ./ 2) .* gamma.((m + 1.5 .- ia) ./ 2))) .* π .* d_alpha
N_kernel = FT.(N_kernel_f64)

DJ = zeros(NR, Ne); for iR in 1:NR, ie in 1:Ne; DJ[iR, ie] = S_RC[iR] * S_e[iR, ie] * Jac[iR, ie]; end
DJ_vec = FT.(reshape(DJ, NPh))
F0l_all = zeros(FT, NPh, n_l); Omega_res = zeros(FT, NPh, n_l)
for i_l in 1:n_l; l = l_arr[i_l]
    F0l_all[:, i_l] .= FT.(reshape((l .* Omega_1 .+ m .* Omega_2) .* FE .+ m .* FL, NPh))
    Omega_res[:, i_l] .= FT.(reshape(l .* Omega_1 .+ m .* Omega_2, NPh))
end

orbit_time = time() - start_time
@printf("Precomputation: %.2f s\n", orbit_time)

function compute_det(omega)
    omega_ct = CT(omega)
    M = zeros(CT, N_alpha, N_alpha)
    for i_l in 1:n_l
        W = W_l_mat[:, :, i_l]; denom = omega_ct .- Omega_res[:, i_l]
        denom[abs.(denom) .< FT(1e-12)] .= CT(1e-12 + 1e-12im)
        weight = DJ_vec .* F0l_all[:, i_l] ./ denom
        M .+= transpose(W) * (weight .* conj.(W))
    end
    M .*= G; M .*= reshape(N_kernel, 1, :)
    return det(Matrix{CT}(I, N_alpha, N_alpha) - M)
end

# ==================== NEWTON-RAPHSON ====================
println("\n" * "=" ^ 60)
println("NEWTON-RAPHSON SEARCH")
println("=" ^ 60)

Omega_p, gamma_val = ref_Omega_p, ref_gamma
@printf("Reference: Ω_p=%.10f, γ=%.10f\n", ref_Omega_p, ref_gamma)
@printf("Initial:   Ω_p=%.6f, γ=%.6f\n\n", Omega_p, gamma_val)

newton_start = time()
for iter in 1:max_iter
    global Omega_p, gamma_val
    omega = m * Omega_p + im * gamma_val; det_val = compute_det(omega)
    F = [real(det_val), imag(det_val)]; res_norm = norm(F)
    @printf("Iter %2d: Ω_p=%.10f γ=%.10f |F|=%.2e\n", iter, Omega_p, gamma_val, res_norm)
    res_norm < newton_tol && (println("\n*** Converged! ***"); break)
    det_dOp = compute_det(m * (Omega_p + delta) + im * gamma_val)
    det_dg = compute_det(m * Omega_p + im * (gamma_val + delta))
    J = hcat(([real(det_dOp), imag(det_dOp)] .- F) ./ delta, ([real(det_dg), imag(det_dg)] .- F) ./ delta)
    dx = J \ F; Omega_p -= dx[1]; gamma_val -= dx[2]
end
newton_time = time() - newton_start

@printf("\n=== Final Result ===\nΩ_p = %.10f\nγ   = %.10f\n", Omega_p, gamma_val)

elapsed = time() - start_time
println("\n" * "=" ^ 60)
@printf("Precomputation: %.2f s\nNewton:         %.2f s\nTotal:          %.2f s\n", orbit_time, newton_time, elapsed)
println("Precision:      ", USE_FLOAT32 ? "Float32" : "Float64")
