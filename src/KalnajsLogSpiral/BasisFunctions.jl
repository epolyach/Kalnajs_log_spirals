# src/KalnajsLogSpiral/BasisFunctions.jl
"""
Basis functions for log-spiral expansion.

Computes:
- W_l(α; J) basis functions via Eq. 41 - COMPLEX VALUED
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
W_l_mat is COMPLEX (this is critical for the eigenvalue problem)
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
    
    # W_l matrices [NPh, N_alpha, n_l] - COMPLEX
    W_l_mat::Array{Complex{T},3}
    
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
function precompute_all(config::KalnajsConfig, model, backend::GPUBackend.GPUType;
                        verbose::Bool=false)
    
    T = Float64  # Pre-GPU computations always use Float64
    
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
        println("Computing W_l basis functions (complex)...")
    end
    
    # Compute W_l basis functions - COMPLEX
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
# W_l Basis Functions (Eq. 41) - COMPLEX
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
    compute_W_l(config, orbit_data, alpha_arr, l_arr, backend; verbose=false) -> Array{Complex{T},3}

Compute W_l basis functions via Eq. 41.

W_l(α; J) = (1/π) ∫ cos(l×w₁ + m×φ_a) × exp(-iα×log(r)) / √r × dw₁

RETURNS COMPLEX ARRAY [NPh, N_alpha, n_l]
"""
function compute_W_l(config::KalnajsConfig, orbit_data::OrbitData{T},
                     alpha_arr::Vector{T}, l_arr::Vector{Int},
                     backend::GPUBackend.GPUType;
                     verbose::Bool=false) where T
    
    NR = config.grid.NR
    Ne = config.grid.Ne
    N_alpha = length(alpha_arr)
    n_l = length(l_arr)
    nwa = size(orbit_data.w1, 1)
    NPh = NR * Ne
    m = config.physics.m
    
    # Allocate output - COMPLEX
    CT = Complex{T}
    W_l = zeros(CT, NR, Ne, N_alpha, n_l)
    
    # CPU implementation
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
                    # THIS IS COMPLEX - exp(-i*alpha*log_r)
                    integrand = cos.(angle_part) .* exp.(-im .* alpha .* log_r) ./ sqrt.(r_vals)
                    
                    # Handle NaN
                    integrand[.!isfinite.(integrand)] .= zero(CT)
                    
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
"""
function gamma_complex(z::Number)
    if z isa AbstractArray
        return gamma_complex.(z)
    end
    
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
    
    if real(z) < 0
        z = -z
    end
    
    z = z - 1
    zh = z + 0.5
    zgh = zh + g
    
    zp = zgh^(zh * 0.5)
    
    ss = 0.0 + 0.0im
    for pp in length(c)-1:-1:1
        ss = ss + c[pp+1] / (z + pp)
    end
    
    sq2pi = 2.5066282746310005024157652848110
    
    f = sq2pi * (c[1] + ss) * (zp * exp(-zgh)) * zp
    
    if z == 0 || z == 1
        f = 1.0 + 0.0im
    end
    
    if real(zz) < 0
        f = -π / (zz * f * sin(π * zz))
    end
    
    if round(real(zz)) == real(zz) && imag(zz) == 0 && real(zz) <= 0
        f = Inf
    end
    
    return f
end


# ============================================================================
# Device-Resident Precomputed Data for GPU
# ============================================================================

"""
    DevicePrecomputed{T,AT}

GPU-resident version of precomputed arrays.
T is the GPU precision (Float32 or Float64).
AT is the array type (CuArray or ROCArray).
W_l_mat is COMPLEX.
"""
struct DevicePrecomputed{T<:AbstractFloat, AT<:AbstractArray}
    m::Int
    G::T
    N_alpha::Int
    n_l::Int
    NPh::Int
    
    # Device arrays - W_l is COMPLEX
    W_l_mat::AT              # [NPh, N_alpha, n_l] - Complex{T}
    DJ_vec::AbstractVector{T}    # [NPh] - kept on host for now
    F0l_all::AbstractMatrix{T}   # [NPh, n_l] - kept on host for now
    Omega_res::AbstractMatrix{T} # [NPh, n_l] - kept on host for now
    N_kernel::AbstractVector{T}  # [N_alpha] - kept on host for now
    d_alpha::AbstractVector{T}   # [N_alpha] - kept on host for now
end

function to_device(precomp::PrecomputedData{T}, gpu_type, Tgpu::Type) where T
    CTgpu = Complex{Tgpu}
    
    # CPU fallback - just convert precision
    if gpu_type == GPUBackend.NONE
        return DevicePrecomputed{Tgpu, Array{CTgpu,3}}(
            precomp.m,
            Tgpu(precomp.G),
            precomp.N_alpha,
            precomp.n_l,
            precomp.NPh,
            CTgpu.(precomp.W_l_mat),
            Tgpu.(precomp.DJ_vec),
            Tgpu.(precomp.F0l_all),
            Tgpu.(precomp.Omega_res),
            Tgpu.(precomp.N_kernel),
            Tgpu.(precomp.d_alpha)
        )
    end
    
    # GPU - use gpu_array_type from GPUBackend
    ArrayType = GPUBackend.gpu_array_type()
    
    # Transfer W_l to device (this is the largest array) - COMPLEX
    W_l_device = ArrayType(CTgpu.(precomp.W_l_mat))
    
    # Other arrays stay on host for now
    return DevicePrecomputed{Tgpu, typeof(W_l_device)}(
        precomp.m,
        Tgpu(precomp.G),
        precomp.N_alpha,
        precomp.n_l,
        precomp.NPh,
        W_l_device,
        Tgpu.(precomp.DJ_vec),
        Tgpu.(precomp.F0l_all),
        Tgpu.(precomp.Omega_res),
        Tgpu.(precomp.N_kernel),
        Tgpu.(precomp.d_alpha)
    )
end
export DevicePrecomputed, to_device
end # module BasisFunctions
