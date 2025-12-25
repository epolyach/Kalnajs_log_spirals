#!/usr/bin/env julia

const HELP_TEXT = """
Kalnajs Log-Spiral Multi-GPU Solver - Standalone Version

Usage: julia --project=. NL_julia_gpu.jl [options]

Options:
  --config=FILE    Configuration file (default: configs/default.toml)
  --gpu=IDS        GPU device IDs (default: 01)
  --threads=N      BLAS threads limit (default: from config)

Examples:
  JULIA_NUM_THREADS=2 julia --project=. NL_julia_gpu.jl --config=configs/default.toml --gpu=01
  JULIA_NUM_THREADS=2 julia --project=. NL_julia_gpu.jl --config=configs/large.toml --gpu=012 --threads=4

Note: Requires JULIA_NUM_THREADS=N where N is the number of GPUs for parallel execution.
"""

start_time = time()

using LinearAlgebra, CUDA, Printf, SpecialFunctions, TOML
CUDA.allowscalar(false)

# ==================== CLI PARSING ====================
function parse_args()
    config_file = "configs/default.toml"
    gpu_str = "01" 
    threads_override = nothing
    
    for arg in ARGS
        if startswith(arg, "--config=")
            config_file = split(arg, "=")[2]
        elseif startswith(arg, "--gpu=")
            gpu_str = split(arg, "=")[2]
        elseif startswith(arg, "--threads=")
            threads_override = parse(Int, split(arg, "=")[2])
        elseif arg == "--help" || arg == "-h"
            println(HELP_TEXT)
            exit(0)
        end
    end
    return config_file, gpu_str, threads_override
end

config_file, gpu_str, threads_override = parse_args()

# ==================== LOAD CONFIG ====================
println("Loading config: $config_file")
config = TOML.parsefile(config_file)

# Apply thread limit
max_threads = get(get(config, "cpu", Dict()), "max_threads", 4)
if threads_override !== nothing
    max_threads = min(threads_override, max_threads)
end
BLAS.set_num_threads(max_threads)

# ==================== PRECISION ====================
const USE_FLOAT32 = !get(config["precision"], "gpu_double_precision", false)
const FT = USE_FLOAT32 ? Float32 : Float64
const CT = USE_FLOAT32 ? ComplexF32 : ComplexF64

# ==================== MULTI-GPU SETUP ====================
gpu_ids = [parse(Int, string(c)) for c in gpu_str]
n_gpus = length(gpu_ids)

println("=" ^ 60)
println("KALNAJS LOG-SPIRAL MULTI-GPU SOLVER") 
println("=" ^ 60)
println("Config:    $config_file")
println("Precision: ", USE_FLOAT32 ? "Float32" : "Float64")
println("BLAS threads: $max_threads")
println("GPUs:      $n_gpus devices - ", [CUDA.name(CuDevice(i)) for i in gpu_ids])

nthreads = Threads.nthreads()
if nthreads < n_gpus
    @warn "Julia started with $nthreads threads but $n_gpus GPUs requested. Set JULIA_NUM_THREADS=$n_gpus for optimal performance."
end
println()

# ==================== PARAMETERS FROM CONFIG ====================
const m = config["physics"]["m"]
const l_min = config["grid"]["l_min"]
const l_max = config["grid"]["l_max"]
const G = FT(config["model"]["G"])
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

const ref_Omega_p = get(config["reference"], "Omega_p", 0.44)
const ref_gamma = get(config["reference"], "gamma", 0.13)

println("Grid: NR=$NR, Ne=$Ne, N_alpha=$N_alpha, l=[$l_min,$l_max], NPh=$NPh")

# ==================== TOOMRE-ZANG MODEL ====================
const L0 = 1.0
const n_zang = config["model"]["n_zang"]
const q1 = config["model"]["q1"]
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

# ==================== CPU PRECOMPUTATION ====================
println("Setting up grids...")
alpha_arr = collect(range(-alpha_max, alpha_max, length=N_alpha))
d_alpha = TrapezoidCoef(alpha_arr)
Rc_ = 10.0 .^ range(log10(Rc_min), log10(Rc_max), length=NR)
S_RC = Rc_ .* TrapezoidCoef(log.(Rc_))
nwai = (nw - 1) ÷ (nwa - 1); Ia = 1:nwai:nw

S_e = zeros(NR, Ne); v_iR = zeros(NR, Ne)
for iR in 1:NR
    v = collect(range(v_min_cfg, 1.0, length=Ne))
    S_e[iR, :] .= SimpsonCoef(v); v_iR[iR, :] .= v
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

# ==================== W_l GPU KERNEL ====================
function w_l_kernel!(W_l_slice, w1, ra, pha, Omega_2, Omega_1, alpha_arr, l_arr, m, π_inv)
    iR = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ie = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    (iR > size(W_l_slice, 1) || ie > size(W_l_slice, 2)) && return

    w1_v = view(w1, :, iR, ie)
    ra_v = view(ra, :, iR, ie)
    pha_v = view(pha, :, iR, ie)
    
    # Check for non-finite values (no broadcasting)
    nwa = length(w1_v)
    for i in 1:nwa
        if !isfinite(w1_v[i]) || !isfinite(ra_v[i])
            return
        end
    end
    
    Om2_Om1 = Omega_2[iR, ie] / Omega_1[iR, ie]
    
    for i_alpha in 1:length(alpha_arr), l_idx in 1:length(l_arr)
        alpha = alpha_arr[i_alpha]
        l = l_arr[l_idx]
        integrand_sum = zero(eltype(W_l_slice))
        
        for iw in 1:nwa
            # Trapezoidal weights inline
            if iw == 1
                Sw1 = (w1_v[2] - w1_v[1]) / 2
            elseif iw == nwa
                Sw1 = (w1_v[nwa] - w1_v[nwa-1]) / 2
            else
                Sw1 = (w1_v[iw+1] - w1_v[iw-1]) / 2
            end
            
            w1_val = w1_v[iw]
            ra_val = ra_v[iw]
            pha_val = pha_v[iw]
            phi_val = Om2_Om1 * w1_val - pha_val
            exp_val = exp(-1im * alpha * log(ra_val))
            cos_val = cos(l * w1_val + m * phi_val)
            integrand_sum += Sw1 * cos_val * exp_val / sqrt(ra_val)
        end
        W_l_slice[iR, ie, i_alpha, l_idx] = integrand_sum * π_inv
    end
    return nothing
end

println("Computing W_l on GPU...")
wl_start = time()
l_split = findfirst(x -> x >= 0, l_arr)
l_neg_count = l_split - 1; l_pos_count = n_l - l_neg_count

W_l_results = Array{Array{CT,3}}(undef, n_gpus)
l_splits = [1:l_neg_count, l_split:n_l]
gpu_ranges = length(gpu_ids) == 1 ? [1:n_l] : l_splits[1:length(gpu_ids)]

Threads.@threads for i in 1:length(gpu_ids)
    gpu_id = gpu_ids[i]; CUDA.device!(gpu_id)
    l_range = gpu_ranges[i]; n_l_gpu = length(l_range)
    
    d_w1 = CuArray(w1); d_ra = CuArray(ra); d_pha = CuArray(pha)
    d_Omega_2 = CuArray(Omega_2); d_Omega_1 = CuArray(Omega_1)
    d_alpha_arr = CuArray(alpha_arr)
    d_l_arr = CuArray(l_arr[l_range])
    d_l_arr = CuArray(l_arr[l_range])
    W_l_gpu = CUDA.zeros(CT, NR, Ne, N_alpha, n_l_gpu)
    
    block_size = (8, 8); grid_size = (cld(NR, 8), cld(Ne, 8))
    
    CUDA.@sync @cuda threads=block_size blocks=grid_size w_l_kernel!(
        W_l_gpu, d_w1, d_ra, d_pha, d_Omega_2, d_Omega_1, d_alpha_arr, d_l_arr, m, FT(1/π)
    )
    
    W_l_results[i] = Array(reshape(W_l_gpu, NPh, N_alpha, n_l_gpu))
end

W_l_mat = cat(W_l_results..., dims=3)
wl_time = time() - wl_start
@printf("W_l computation: %.2f s\n", wl_time)

# ==================== GPU MEMORY TRANSFER ====================
println("Transferring to GPU...")
gpus_data = [Dict() for _ in 1:n_gpus]

Threads.@threads for i in 1:n_gpus
    gpu_id = gpu_ids[i]; CUDA.device!(gpu_id)
    gpus_data[i]["W_l_mat"] = CuArray(W_l_mat)
    gpus_data[i]["DJ_vec"] = CuArray(DJ_vec)
    gpus_data[i]["F0l_all"] = CuArray(F0l_all)
    gpus_data[i]["Omega_res"] = CuArray(Omega_res)
    gpus_data[i]["N_kernel"] = CuArray(N_kernel)
    gpus_data[i]["eyeN"] = CuArray(Matrix{CT}(I, N_alpha, N_alpha))
end

function compute_det_multi(omega)
    omega_ct = CT(omega)
    M_parts = Vector{Matrix{CT}}(undef, n_gpus)
    
    Threads.@threads for i in 1:n_gpus
        gpu_id = gpu_ids[i]; CUDA.device!(gpu_id)
        gpu_data = gpus_data[i]
        M_part = CUDA.zeros(CT, N_alpha, N_alpha)
        
        for i_l in 1:n_l
            W = view(gpu_data["W_l_mat"], :, :, i_l)
            denom = omega_ct .- view(gpu_data["Omega_res"], :, i_l)
            denom = ifelse.(abs.(denom) .< FT(1e-12), CT(1e-12 + 1e-12im), denom)
            weight = gpu_data["DJ_vec"] .* view(gpu_data["F0l_all"], :, i_l) ./ denom
            M_part .+= transpose(W) * (weight .* conj.(W))
        end
        
        M_parts[i] = Array(M_part)
    end
    
    M = sum(M_parts); M .*= G; M .*= reshape(N_kernel, 1, :)
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
    omega = m * Omega_p + im * gamma_val; det_val = compute_det_multi(omega)
    F = [real(det_val), imag(det_val)]; res_norm = norm(F)
    @printf("Iter %2d: Ω_p=%.10f γ=%.10f |F|=%.2e\n", iter, Omega_p, gamma_val, res_norm)
    res_norm < newton_tol && (println("\n*** Converged! ***"); break)
    det_dOp = compute_det_multi(m * (Omega_p + delta) + im * gamma_val)
    det_dg = compute_det_multi(m * Omega_p + im * (gamma_val + delta))
    J = hcat(([real(det_dOp), imag(det_dOp)] .- F) ./ delta, ([real(det_dg), imag(det_dg)] .- F) ./ delta)
    dx = J \ F; Omega_p -= dx[1]; gamma_val -= dx[2]
end
newton_time = time() - newton_start

@printf("\n=== Final Result ===\nΩ_p = %.10f\nγ   = %.10f\n", Omega_p, gamma_val)

elapsed = time() - start_time
println("\n" * "=" ^ 60)
@printf("Precomputation: %.2f s\nW_l GPU:        %.2f s\nNewton:         %.2f s\nTotal:          %.2f s\n", orbit_time, wl_time, newton_time, elapsed)
println("Precision:      ", USE_FLOAT32 ? "Float32" : "Float64")
