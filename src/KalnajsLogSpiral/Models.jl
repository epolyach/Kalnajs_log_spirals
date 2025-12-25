# src/KalnajsLogSpiral/Models.jl
"""
Toomre-Zang model for Kalnajs log-spiral eigenvalue calculations.

This is adapted from ../PME/src/models/Toomre.jl, keeping the same physics.
"""
module Models

using SpecialFunctions
using ..Configuration

export ToomreZangModel, create_toomre_zang_model
export potential, potential_derivative, rotation_frequency, epicyclic_frequency
export surface_density, distribution_function, taper

# ============================================================================
# Toomre-Zang Model Structure
# ============================================================================

"""
Toomre-Zang galactic disk model with logarithmic potential.
All quantities stored as Float64 (pre-GPU computations).
"""
struct ToomreZangModel{T<:AbstractFloat}
    L0::T
    n_zang::Int
    q1::Int
    q_zang::Int
    sigma_r0::T
    G::T
    DF_const::T
end

"""
    create_toomre_zang_model(config::KalnajsConfig) -> ToomreZangModel{Float64}

Create a Toomre-Zang model from configuration.
Always returns Float64 model (pre-GPU computations are not performance-critical).
"""
function create_toomre_zang_model(config::KalnajsConfig)
    T = Float64  # Pre-GPU is always Float64
    
    L0 = T(config.model.L0)
    n_zang = config.model.n_zang
    q1 = config.model.q1
    G = T(config.physics.G)
    
    # Derived parameters
    sigma_r0 = T(1.0) / sqrt(T(q1))
    q_zang = q1 - 1
    
    # Normalization constant for distribution function
    DF_const = T(1.0) / (T(2.0) * T(π) * T(2.0)^(q_zang/2) * sqrt(T(π)) * 
                         gamma(T(q_zang + 1)/2) * sigma_r0^(q_zang + 2))
    
    return ToomreZangModel{T}(L0, n_zang, q1, q_zang, sigma_r0, G, DF_const)
end

# ============================================================================
# Potential and Kinematics
# ============================================================================

"""Logarithmic potential: V(r) = log(r)"""
potential(model::ToomreZangModel{T}, r::Real) where T = log(T(r))

"""Potential derivative: dV/dr = 1/r"""
potential_derivative(model::ToomreZangModel{T}, r::Real) where T = T(1) / T(r)

"""Rotation frequency: Ω(r) = 1/r (flat rotation curve)"""
rotation_frequency(model::ToomreZangModel{T}, r::Real) where T = T(1) / T(r)

"""Epicyclic frequency: κ(r) = √2/r"""
epicyclic_frequency(model::ToomreZangModel{T}, r::Real) where T = sqrt(T(2)) / T(r)

"""Surface density: Σ(r) = 1/(2πr)"""
surface_density(model::ToomreZangModel{T}, r::Real) where T = T(1) / (T(2) * T(π) * T(r))

# ============================================================================
# Taper Function
# ============================================================================

"""
    taper(model::ToomreZangModel, L::Real) -> Real

Zang taper factor: [1 + (L₀/L)^n]^{-1}

Special cases:
- n_zang = 0: taper = 1 (no taper)
- |L| < 1e-12: taper = 0 (exclude radial orbits)
"""
function taper(model::ToomreZangModel{T}, L::Real) where T
    if model.n_zang == 0
        return T(1)
    end
    
    L_abs = abs(T(L))
    if L_abs < T(1e-12)
        return T(0)
    end
    
    return T(1) / (T(1) + (model.L0 / L_abs)^model.n_zang)
end

"""
    taper_derivative(model::ToomreZangModel, L::Real) -> Real

Derivative of taper with respect to L.
"""
function taper_derivative(model::ToomreZangModel{T}, L::Real) where T
    if model.n_zang == 0
        return T(0)
    end
    
    L_abs = abs(T(L))
    if L_abs < T(1e-12)
        return T(0)
    end
    
    sign_L = sign(T(L))
    ratio = model.L0 / L_abs
    taper_val = taper(model, L)
    
    return model.n_zang * ratio^model.n_zang * taper_val^2 * sign_L / L_abs
end

# ============================================================================
# Distribution Function
# ============================================================================

"""
    distribution_function(model::ToomreZangModel, E::Real, L::Real) -> Real

Distribution function: f₀(E, L) = C exp(-E/σ²) |L|^q taper(L)

Returns 0 for |L| < 1e-12 or non-finite results.
"""
function distribution_function(model::ToomreZangModel{T}, E::Real, L::Real) where T
    L_abs = abs(T(L))
    if L_abs < T(1e-12)
        return T(0)
    end
    
    E_val = T(E)
    sigma_sq = model.sigma_r0^2
    taper_val = taper(model, L)
    
    result = model.DF_const * exp(-E_val / sigma_sq) * L_abs^model.q_zang * taper_val
    
    return isfinite(result) ? result : T(0)
end

"""∂f₀/∂E"""
function DF_E(model::ToomreZangModel{T}, E::Real, L::Real) where T
    return -distribution_function(model, E, L) / model.sigma_r0^2
end

"""∂f₀/∂L - PME/MATLAB formula: DF_L = DF * (q_zang + n_zang - n_zang*taper) / |L|"""
function DF_L(model::ToomreZangModel{T}, E::Real, L::Real) where T
    L_abs = abs(T(L))
    if L_abs < T(1e-12)
        return T(0)
    end
    
    f0 = distribution_function(model, E, L)
    taper_val = taper(model, L)
    
    # PME formula uses |L| in denominator
    return f0 * (model.q_zang + model.n_zang - model.n_zang * taper_val) / L_abs
end

# ============================================================================
# Helper Functions for Orbit Calculations
# ============================================================================

"""
Compute energy: E = ½(dr/dt)² + V_eff(r)
For circular orbits (r1 ≈ r2): E ≈ r × dV/dr / 2 + V(r)
"""
function compute_energy(model::ToomreZangModel{T}, r1::Real, r2::Real) where T
    r_avg = (T(r1) + T(r2)) / 2
    
    if abs(T(r1) - T(r2)) / r_avg < T(1e-6)
        # Circular limit
        return r_avg * potential_derivative(model, r_avg) / 2 + potential(model, r_avg)
    else
        # General case (would need full orbit integration)
        return r_avg * potential_derivative(model, r_avg) / 2 + potential(model, r_avg)
    end
end

"""
Compute angular momentum squared from energy and apocenter.
"""
function compute_L_squared(model::ToomreZangModel{T}, E::Real, r2::Real) where T
    E_val = T(E)
    r2_val = T(r2)
    
    L_sq = r2_val * (r2_val * potential_derivative(model, r2_val) - 2 * (E_val - potential(model, r2_val)))
    
    return max(L_sq, T(0))
end

"""
Compute angular momentum with sign.
"""
function compute_L(model::ToomreZangModel{T}, E::Real, r2::Real, sign_L::Real) where T
    L_sq = compute_L_squared(model, E, r2)
    return T(sign_L) * sqrt(L_sq)
end

# Short aliases
V(model::ToomreZangModel, r) = potential(model, r)
dV(model::ToomreZangModel, r) = potential_derivative(model, r)
Omega(model::ToomreZangModel, r) = rotation_frequency(model, r)
kappa(model::ToomreZangModel, r) = epicyclic_frequency(model, r)
Sigma(model::ToomreZangModel, r) = surface_density(model, r)
DF(model::ToomreZangModel, E, L) = distribution_function(model, E, L)

# Alias for compatibility with OrbitIntegration
df_energy_derivative(model::ToomreZangModel, E, L) = DF_E(model, E, L)
df_angular_derivative(model::ToomreZangModel, E, L) = DF_L(model, E, L)

end # module Models
