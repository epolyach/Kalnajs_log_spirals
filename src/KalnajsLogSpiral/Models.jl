# src/KalnajsLogSpiral/Models.jl
"""
Toomre-Zang galactic disk model.

Implements the logarithmic potential model with Zang taper:
- V(r) = log(r)
- Ω(r) = 1/r (flat rotation curve)
- κ(r) = √2/r
- DF(E,L) = C × exp(-E/σ²) × L^q × taper(L)

Reference: Zang (1976) PhD thesis, Toomre (1981)
"""
module Models

using SpecialFunctions
using ..Configuration

export ToomreZangModel, create_toomre_zang_model

# ============================================================================
# Model Structure
# ============================================================================

"""
Toomre-Zang model with precomputed constants
"""
struct ToomreZangModel{T<:AbstractFloat}
    # Model parameters
    L0::T           # Angular momentum scale
    n_zang::Int     # Zang taper exponent
    q1::Int         # Velocity dispersion parameter
    q_zang::Int     # = q1 - 1
    sigma_r0::T     # Radial velocity dispersion = 1/√q1
    G::T            # Gravitational constant
    
    # Normalization constant
    DF_const::T
end

"""
    create_toomre_zang_model(config::KalnajsConfig) -> ToomreZangModel

Create Toomre-Zang model from configuration.
"""
function create_toomre_zang_model(config::KalnajsConfig)
    T = config.gpu.precision_double ? Float64 : Float32
    
    L0 = T(config.model.L0)
    n_zang = config.model.n_zang
    q1 = config.model.q1
    q_zang = q1 - 1
    sigma_r0 = T(1.0 / sqrt(q1))
    G = T(config.physics.G)
    
    # Normalization constant: 1 / (2π × 2^(q/2) × √π × Γ((q+1)/2) × σ^(q+2))
    DF_const = T(1.0 / (2π * 2^(q_zang/2) * sqrt(π) * 
                        gamma((q_zang + 1)/2) * sigma_r0^(q_zang + 2)))
    
    return ToomreZangModel{T}(L0, n_zang, q1, q_zang, sigma_r0, G, DF_const)
end

# ============================================================================
# Potential and Kinematic Functions
# ============================================================================

"""
Logarithmic potential: V(r) = log(r)
"""
@inline function potential(model::ToomreZangModel{T}, r::T) where T
    return log(r)
end

"""
Potential derivative: dV/dr = 1/r
"""
@inline function potential_derivative(model::ToomreZangModel{T}, r::T) where T
    return one(T) / r
end

"""
Rotation frequency: Ω(r) = 1/r (flat rotation curve)
"""
@inline function rotation_frequency(model::ToomreZangModel{T}, r::T) where T
    return one(T) / r
end

"""
Epicyclic frequency: κ(r) = √2/r
"""
@inline function epicyclic_frequency(model::ToomreZangModel{T}, r::T) where T
    return T(sqrt(T(2))) / r
end

"""
Surface density: Σ(r) = 1/(2πr)
"""
@inline function surface_density(model::ToomreZangModel{T}, r::T) where T
    return one(T) / (T(2π) * r)
end

# ============================================================================
# Distribution Function
# ============================================================================

"""
Zang taper function: [1 + (L₀/L)^n]^{-1}

For n_zang = 0, returns 1 (no taper)
For n_zang > 0, suppresses small angular momentum
"""
@inline function taper(model::ToomreZangModel{T}, L::T) where T
    if model.n_zang == 0
        return one(T)
    end
    
    L_abs = abs(L)
    if L_abs < T(1e-12)
        return zero(T)
    end
    
    return one(T) / (one(T) + (model.L0 / L_abs)^model.n_zang)
end

"""
Derivative of taper function with respect to L
"""
@inline function taper_derivative(model::ToomreZangModel{T}, L::T) where T
    if model.n_zang == 0
        return zero(T)
    end
    
    L_abs = abs(L)
    if L_abs < T(1e-12)
        return zero(T)
    end
    
    n = model.n_zang
    ratio = model.L0 / L_abs
    t = one(T) / (one(T) + ratio^n)
    
    # d/dL [1/(1 + (L0/L)^n)] = n * (L0/L)^n / (L * (1 + (L0/L)^n)^2)
    return n * ratio^n * t^2 / L_abs * sign(L)
end

"""
Distribution function: DF(E, L)

DF(E,L) = C × exp(-E/σ²) × |L|^q × taper(L)

Returns 0 for:
- Non-finite results
- |L| < 1e-12 (radial orbits)
"""
@inline function distribution_function(model::ToomreZangModel{T}, E::T, L::T) where T
    L_abs = abs(L)
    
    # Guard against radial orbits
    if L_abs < T(1e-12)
        return zero(T)
    end
    
    exp_factor = exp(-E / model.sigma_r0^2)
    L_power = L_abs^model.q_zang
    taper_factor = taper(model, L)
    
    result = model.DF_const * exp_factor * L_power * taper_factor
    
    # Guard against numerical issues
    if !isfinite(result)
        return zero(T)
    end
    
    return result
end

"""
Energy derivative of distribution function: ∂DF/∂E

∂DF/∂E = -DF(E,L) / σ²
"""
@inline function df_energy_derivative(model::ToomreZangModel{T}, E::T, L::T) where T
    df_val = distribution_function(model, E, L)
    return -df_val / model.sigma_r0^2
end

"""
Angular momentum derivative of distribution function: ∂DF/∂L

∂DF/∂L = DF(E,L) × (q + n - n×taper(L)) / L
"""
@inline function df_angular_derivative(model::ToomreZangModel{T}, E::T, L::T) where T
    L_abs = abs(L)
    
    if L_abs < T(1e-12)
        return zero(T)
    end
    
    df_val = distribution_function(model, E, L)
    if df_val == zero(T)
        return zero(T)
    end
    
    taper_factor = taper(model, L)
    n = model.n_zang
    q = model.q_zang
    
    # Coefficient: (q + n - n*taper) / |L| × sign(L)
    coeff = (q + n - n * taper_factor) / L_abs
    
    return df_val * coeff * sign(L)
end

# ============================================================================
# Energy and Angular Momentum from Orbital Elements
# ============================================================================

"""
Compute energy from pericenter and apocenter radii.

E = (V(r2)×r2² - V(r1)×r1²) / (r2² - r1²)

For circular orbits (r1 ≈ r2): E = r×dV/2 + V(r)
"""
function compute_energy(model::ToomreZangModel{T}, r1::T, r2::T) where T
    if abs(r2 - r1) < T(1e-12) * max(r1, r2)
        # Circular orbit limit
        r = (r1 + r2) / 2
        return r * potential_derivative(model, r) / 2 + potential(model, r)
    end
    
    V1 = potential(model, r1)
    V2 = potential(model, r2)
    
    return (V2 * r2^2 - V1 * r1^2) / (r2^2 - r1^2)
end

"""
Compute angular momentum squared from energy and apocenter.

L² = 2×(E - V(r2))×r2²
"""
function compute_L_squared(model::ToomreZangModel{T}, E::T, r2::T) where T
    V2 = potential(model, r2)
    L2 = T(2) * (E - V2) * r2^2
    return max(L2, zero(T))  # Ensure non-negative
end

"""
Compute angular momentum from energy and apocenter.
"""
function compute_L(model::ToomreZangModel{T}, E::T, r2::T, sign_L::T=one(T)) where T
    L2 = compute_L_squared(model, E, r2)
    return sqrt(L2) * sign_L
end

# ============================================================================
# Convenience Functions
# ============================================================================

# Allow calling with Float64 inputs regardless of model precision
for func in [:potential, :potential_derivative, :rotation_frequency, 
             :epicyclic_frequency, :surface_density, :taper, 
             :distribution_function, :df_energy_derivative, :df_angular_derivative]
    @eval begin
        function $func(model::ToomreZangModel{T}, args::Real...) where T
            return $func(model, T.(args)...)
        end
    end
end

# Short aliases
const V = potential
const dV = potential_derivative
const Omega = rotation_frequency
const kappa = epicyclic_frequency
const Sigma = surface_density
const DF = distribution_function
const DF_E = df_energy_derivative
const DF_L = df_angular_derivative

export V, dV, Omega, kappa, Sigma, DF, DF_E, DF_L
export potential, potential_derivative, rotation_frequency, epicyclic_frequency
export surface_density, taper, distribution_function
export df_energy_derivative, df_angular_derivative
export compute_energy, compute_L, compute_L_squared

end # module Models
