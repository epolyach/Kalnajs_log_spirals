# src/KalnajsLogSpiral/BasisFunctions.jl
"""
Basis functions for log-spiral expansion.

Computes:
- W_l(α; J) basis functions via Eq. 41
- N_m(α) kernel via Eq. 46

With GPU acceleration using KernelAbstractions.jl
"""
module BasisFunctions

using KernelAbstractions
using SpecialFunctions
using ..Configuration
using ..GPUBackend
using ..OrbitIntegration

export PrecomputedData, precompute_all
export compute_W_l, compute_N_kernel

# ============================================================================
# Precomputed Data Structure
# ============================================================================

"""
All ω-independent precomputed quantities
"""
struct PrecomputedData{T<:AbstractFloat}
    # Grid parameters
    m::Int
    G::T
    N_alpha::Int
    n_l::Int
    l_arr::Vector{Int}
    NPh::Int
    
    # Alpha grid
    alpha_arr::Vector{T}
    d_alpha::Vector{T}
    
    # Orbit data
    orbit_data::OrbitData{T}
    
    # Integration weights: DJ = Jacobian × S_RC × S_e
    DJ_vec::Vector{T}
    
    # F_{0,l}(J) = (l×Ω₁ + m×Ω₂) × ∂F₀/∂E + m × ∂F₀/∂L  [NPh, n_l]
    F0l_all::Matrix{T}
    
    # Resonance frequencies: l×Ω₁ + m×Ω₂  [NPh, n_l]
    Omega_res::Matrix{T}
    
    # W_l matrices [NPh, N_alpha, n_l]
    W_l_mat::Array{T,3}
    
    # N_m(α) kernel [N_alpha]
    N_kernel::Vector{T}
end

# ============================================================================
# Main Precomputation Function
# ============================================================================

"""
    precompute_all(config::KalnajsConfig, model, backend; verbose=false) -> PrecomputedData

Precompute all ω-independent quantities.

This is the Julia equivalent of NL_precompute.m
"""
function precompute_all(config::KalnajsConfig, model, backend::GPUBackend.GPUBackendType;
                        verbose::Bool=false)
    
    T = config.gpu.precision_double ? Float64 : Float32
    
    # Extract parameters
    m = config.physics.m
    G = T(config.physics.G)
    N_alpha = config.grid.N_alpha
    alpha_max = T(config.grid.alpha_max)
    l_min = config.grid.l_min
    l_max = config.grid.l_max
    n_l = l_max - l_min + 1
    l_arr = collect(l_min:l_max)
    
    NR = config.grid.NR
    Ne = config.grid.Ne
    NPh = NR * Ne
    
    if verbose
        println("Computing grids...")
    end
    
    # Compute grids and orbits
    grids = OrbitIntegration.compute_grids(config, model)
    
    if verbose
        println("Computing orbits...")
    end
    
    orbit_data = OrbitIntegration.compute_orbits(config, model, grids; verbose=verbose)
    
    # Alpha grid
    alpha_arr = T.(range(-alpha_max, alpha_max, length=N_alpha))
    d_alpha = trapezoid_coef(alpha_arr)
    
    if verbose
        println("Computing W_l basis functions...")
    end
    
    # Compute W_l basis functions
    W_l_mat = compute_W_l(config, orbit_data, alpha_arr, l_arr, backend; verbose=verbose)
    
    if verbose
        println("Computing N_m(α) kernel...")
    end
    
    # Compute N_m(α) kernel
    N_kernel = compute_N_kernel(m, alpha_arr, d_alpha)
    
    if verbose
        println("Computing ω-independent quantities...")
    end
    
    # Integration weights: DJ = Jacobian × S_RC × S_e
    S_RC = orbit_data.grids.S_RC
    S_e = orbit_data.grids.S_e
    DJ = similar(orbit_data.jacobian)
    for iR in 1:NR
        for ie in 1:Ne
            DJ[iR, ie] = S_RC[iR] * S_e[iR, ie] * orbit_data.jacobian[iR, ie]
        end
    end
    DJ_vec = reshape(DJ, NPh)
    
    # F_{0,l}(J) = (l×Ω₁ + m×Ω₂) × ∂F₀/∂E + m × ∂F₀/∂L
    F0l_all = zeros(T, NPh, n_l)
    Omega_1_vec = reshape(orbit_data.Omega_1, NPh)
    Omega_2_vec = reshape(orbit_data.Omega_2, NPh)
    FE_vec = reshape(orbit_data.FE, NPh)
    FL_vec = reshape(orbit_data.FL, NPh)
    
    for (i_l, l) in enumerate(l_arr)
        F0l_all[:, i_l] .= (l .* Omega_1_vec .+ m .* Omega_2_vec) .* FE_vec .+ m .* FL_vec
    end
    
    # Resonance frequencies: l×Ω₁ + m×Ω₂
    Omega_res = zeros(T, NPh, n_l)
    for (i_l, l) in enumerate(l_arr)
        Omega_res[:, i_l] .= l .* Omega_1_vec .+ m .* Omega_2_vec
    end
    
    if verbose
        println("Precomputation complete.")
    end
    
    return PrecomputedData{T}(
        m, G, N_alpha, n_l, l_arr, NPh,
        alpha_arr, d_alpha,
        orbit_data,
        DJ_vec, F0l_all, Omega_res, W_l_mat, N_kernel
    )
end

# ============================================================================
# W_l Basis Functions (Eq. 41)
# ============================================================================

"""
Compute trapezoidal integration coefficients
"""
function trapezoid_coef(x::AbstractVector{T}) where T
    n = length(x)
    S = zeros(T, n)
    
    if n >= 2
        S[1] = (x[2] - x[1]) / 2
        S[end] = (x[end] - x[end-1]) / 2
        
        for i in 2:n-1
            S[i] = (x[i+1] - x[i-1]) / 2
        end
    end
    
    return S
end

"""
    compute_W_l(config, orbit_data, alpha_arr, l_arr, backend; verbose=false) -> Array{T,3}

Compute W_l basis functions via Eq. 41.

W_l(α; J) = (1/π) ∫ cos(l×w₁ + m×φ_a) × exp(-iα×log(r)) / √r × dw₁

Returns array [NPh, N_alpha, n_l]
"""
function compute_W_l(config::KalnajsConfig, orbit_data::OrbitData{T},
                     alpha_arr::Vector{T}, l_arr::Vector{Int},
                     backend::GPUBackend.GPUBackendType;
                     verbose::Bool=false) where T
    
    NR = config.grid.NR
    Ne = config.grid.Ne
    N_alpha = length(alpha_arr)
    n_l = length(l_arr)
    nwa = size(orbit_data.w1, 1)
    NPh = NR * Ne
    m = config.physics.m
    
    # Allocate output
    W_l = zeros(T, NR, Ne, N_alpha, n_l)
    
    # CPU implementation (GPU kernel would be similar but parallelized)
    for iR in 1:NR
        for ie in 1:Ne
            w1_vals = orbit_data.w1[:, iR, ie]
            r_vals = orbit_data.ra[:, iR, ie]
            theta_vals = orbit_data.pha[:, iR, ie]
            
            # Skip invalid orbits
            if any(.!isfinite.(r_vals)) || any(.!isfinite.(theta_vals)) || any(.!isfinite.(w1_vals))
                continue
            end
            
            # Integration weights for w1
            Sw1 = trapezoid_coef(w1_vals)
            
            # Ω₂/Ω₁ ratio
            Om2_Om1 = orbit_data.Omega_2[iR, ie] / orbit_data.Omega_1[iR, ie]
            
            # φ_a = (Ω₂/Ω₁)×w₁ - θ
            phi_vals = Om2_Om1 .* w1_vals .- theta_vals
            
            for (i_l, l) in enumerate(l_arr)
                for (i_alpha, alpha) in enumerate(alpha_arr)
                    # angle_part = l×w₁ + m×φ_a
                    angle_part = l .* w1_vals .+ m .* phi_vals
                    log_r = log.(r_vals)
                    
                    # integrand = cos(angle) × exp(-iα×log(r)) / √r
                    # For real part: cos(angle) × cos(α×log(r)) / √r
                    integrand = cos.(angle_part) .* cos.(alpha .* log_r) ./ sqrt.(r_vals)
                    
                    # Handle NaN
                    integrand[.!isfinite.(integrand)] .= zero(T)
                    
                    W_l[iR, ie, i_alpha, i_l] = sum(Sw1 .* integrand) / T(π)
                end
            end
        end
        
        if verbose && mod(iR, 20) == 0
            print(".")
        end
    end
    
    if verbose
        println(" done.")
    end
    
    # Reshape to [NPh, N_alpha, n_l]
    return reshape(W_l, NPh, N_alpha, n_l)
end

# ============================================================================
# GPU Kernel for W_l (for future optimization)
# ============================================================================

"""
GPU kernel for W_l computation (KernelAbstractions.jl style)
"""
@kernel function compute_W_l_kernel!(W_l, @Const(w1), @Const(ra), @Const(pha),
                                     @Const(Omega_1), @Const(Omega_2),
                                     @Const(alpha_arr), @Const(l_arr),
                                     m::Int, NR::Int, Ne::Int, N_alpha::Int, n_l::Int, nwa::Int)
    
    # Get global thread index
    idx = @index(Global)
    
    # Decode indices: idx → (iR, ie, i_alpha, i_l)
    total = NR * Ne * N_alpha * n_l
    if idx > total
        return
    end
    
    idx_temp = idx - 1
    i_l = (idx_temp % n_l) + 1
    idx_temp = idx_temp ÷ n_l
    i_alpha = (idx_temp % N_alpha) + 1
    idx_temp = idx_temp ÷ N_alpha
    ie = (idx_temp % Ne) + 1
    iR = (idx_temp ÷ Ne) + 1
    
    # Get parameters
    alpha = alpha_arr[i_alpha]
    l = l_arr[i_l]
    Om2_Om1 = Omega_2[iR, ie] / Omega_1[iR, ie]
    
    # Compute integral
    T = eltype(W_l)
    result = zero(T)
    
    for iw in 1:nwa
        w1_val = w1[iw, iR, ie]
        r_val = ra[iw, iR, ie]
        theta_val = pha[iw, iR, ie]
        
        if !isfinite(r_val) || !isfinite(theta_val) || !isfinite(w1_val)
            continue
        end
        
        # Trapezoidal weight (simplified)
        dw = if iw == 1
            (w1[2, iR, ie] - w1[1, iR, ie]) / T(2)
        elseif iw == nwa
            (w1[nwa, iR, ie] - w1[nwa-1, iR, ie]) / T(2)
        else
            (w1[iw+1, iR, ie] - w1[iw-1, iR, ie]) / T(2)
        end
        
        phi_a = Om2_Om1 * w1_val - theta_val
        angle_part = l * w1_val + m * phi_a
        log_r = log(r_val)
        
        integrand = cos(angle_part) * cos(alpha * log_r) / sqrt(r_val)
        result += dw * integrand
    end
    
    W_l[iR, ie, i_alpha, i_l] = result / T(π)
end

# ============================================================================
# N_m(α) Kernel (Eq. 46)
# ============================================================================

"""
    compute_N_kernel(m::Int, alpha_arr::Vector{T}, d_alpha::Vector{T}) -> Vector{T}

Compute N_m(α) kernel via Eq. 46.

N_m(α) = Γ((m+1/2+iα)/2) × Γ((m+1/2-iα)/2) / (Γ((m+3/2+iα)/2) × Γ((m+3/2-iα)/2)) × π × dα

This uses the complex gamma function for the calculation.
"""
function compute_N_kernel(m::Int, alpha_arr::Vector{T}, d_alpha::Vector{T}) where T
    N_alpha = length(alpha_arr)
    N_kernel = zeros(T, N_alpha)
    
    for (i, alpha) in enumerate(alpha_arr)
        ia = im * alpha
        
        z1 = (m + T(0.5) + ia) / 2
        z2 = (m + T(0.5) - ia) / 2
        z3 = (m + T(1.5) + ia) / 2
        z4 = (m + T(1.5) - ia) / 2
        
        # Complex gamma function
        g1 = gamma_complex(z1)
        g2 = gamma_complex(z2)
        g3 = gamma_complex(z3)
        g4 = gamma_complex(z4)
        
        ratio = (g1 * g2) / (g3 * g4)
        N_kernel[i] = real(ratio) * T(π) * d_alpha[i]
    end
    
    return N_kernel
end

"""
Complex gamma function using Lanczos approximation.

Translated from gamma_complex.m (Godfrey implementation).
Valid in entire complex plane with 15 significant digits on real axis.
"""
function gamma_complex(z::Number)
    # Handle array input
    if z isa AbstractArray
        return gamma_complex.(z)
    end
    
    # Lanczos coefficients (g=607/128)
    g = 607.0 / 128.0
    
    c = [0.99999999999999709182,
         57.156235665862923517,
         -59.597960355475491248,
         14.136097974741747174,
         -0.49191381609762019978,
         0.33994649984811888699e-4,
         0.46523628927048575665e-4,
         -0.98374475304879564677e-4,
         0.15808870322491248884e-3,
         -0.21026444172410488319e-3,
         0.21743961811521264320e-3,
         -0.16431810653676389022e-3,
         0.84418223983852743293e-4,
         -0.26190838401581408670e-4,
         0.36899182659531622704e-5]
    
    zz = z
    
    # Handle negative real part
    if real(z) < 0
        z = -z
    end
    
    z = z - 1
    zh = z + 0.5
    zgh = zh + g
    
    # Avoid overflow: zgh^(zh*0.5)
    zp = zgh^(zh * 0.5)
    
    # Sum the series
    ss = 0.0 + 0.0im
    for pp in length(c)-1:-1:1
        ss = ss + c[pp+1] / (z + pp)
    end
    
    # sqrt(2π)
    sq2pi = 2.5066282746310005024157652848110
    
    f = sq2pi * (c[1] + ss) * (zp * exp(-zgh)) * zp
    
    # Handle z=0 or z=1
    if z == 0 || z == 1
        f = 1.0 + 0.0im
    end
    
    # Adjust for negative real parts
    if real(zz) < 0
        f = -π / (zz * f * sin(π * zz))
    end
    
    # Adjust for negative poles
    if round(real(zz)) == real(zz) && imag(zz) == 0 && real(zz) <= 0
        f = Inf
    end
    
    return f
end

end # module BasisFunctions
