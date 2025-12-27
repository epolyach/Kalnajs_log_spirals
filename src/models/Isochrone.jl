# src/models/Isochrone.jl
"""
Implements the Kalnajs isochrone model with JH and ZH tapers for 
central angular momentum regions, providing clean, lapidary functions
for potential, kinematics, and distribution functions.
"""
module Isochrone

using SpecialFunctions
using QuadGK
using Interpolations
using Interpolations: interpolate, Gridded, Linear, BSpline, Cubic, linear_interpolation, scale
using Dierckx  # For spline derivatives
using ..AbstractModel
using ..Configuration: KalnajsConfig
using Printf

export IsochroneModel, IsochroneJHModel, IsochroneZHModel, IsochroneTanhModel, IsochroneTaperExpModel, create_isochrone_model, create_isochrone_jh_model, create_isochrone_zh_model, create_isochrone_tanh_model, create_isochrone_taper_exp_model

"""
Isochrone model with JH (Jalali & Hunter 2005) taper
"""
struct IsochroneJHModel <: AbstractGalacticModel end

"""
Isochrone model with ZH (Zang & Hohl 1978) taper
"""
struct IsochroneZHModel <: AbstractGalacticModel end

"""
Isochrone model with Tanh taper
"""
struct IsochroneTanhModel <: AbstractGalacticModel end

"""
Isochrone model with exponential taper
"""
struct IsochroneTaperExpModel <: AbstractGalacticModel end

# Plain Isochrone model (no taper)
struct IsochroneModel <: AbstractGalacticModel end

"""
    create_isochrone_model(config::KalnajsConfig) -> ModelResults
    
Create plain (untapered) isochrone model from configuration.
Unidirectional formulation (L ≥ 0) similar to Miyamoto model.
"""
function create_isochrone_model(config::KalnajsConfig)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity
    
    return setup_model(IsochroneModel, mk; unit_mass, unit_length, selfgravity)
end

"""
    setup_model(::Type{IsochroneModel}, mk; kwargs...) -> ModelResults
    
Initialize plain (untapered) isochrone model.
Unidirectional DF (L ≥ 0 only). Uses Kalnajs g-function normalization.
"""
function setup_model(::Type{IsochroneModel}, mk::Int; 
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0)
    
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Isochrone model (plain, no taper) (mk=$mk, selfgravity=$selfgravity)")
    end
    
    # Compute g function via numerical integration
    g_interp = compute_g_function(mk)
    
    # Compute sigma_u function for velocity dispersion
    ppu_interp = compute_sigma_u_function(mk)
    
    # Core kinematic functions (dimensionless isochrone)
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt( sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0)) 
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))
    
    # Surface density
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)
    
    # Velocity dispersion
    sigma_u(r) = sqrt( ppu_interp(-r.*V(r)) ./ r.^(mk+1) ./ Sigma_d(r) )
   
    # Create distribution functions (unidirectional, L ≥ 0)
    DF, DF_dE, DF_dL = create_plain_distribution_functions(mk, g_interp)
    
    # Package parameters
    params = Dict{String,Any}(
        "mk" => mk, "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "DF_type" => 0, "taper_type" => "None"
    )
    
    helpers = (
        model_name = "Kalnajs Isochrone - Plain (no taper)",
        title = "Plain Isochrone mk=$mk",
        Jr = (E, h) -> 1.0/sqrt(-2.0*E) - (h + sqrt(h^2 + 4.0))/2.0
    )
    
    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers, "Isochrone", params
    )
end


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

"""
    create_isochrone_jh_model(config::KalnajsConfig) -> ModelResults
    
Create JH-tapered isochrone model from configuration.
"""
function create_isochrone_jh_model(config::KalnajsConfig)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity
    
    return setup_model(IsochroneJHModel, mk; unit_mass, unit_length, selfgravity)
end

"""
    create_isochrone_zh_model(config::KalnajsConfig) -> ModelResults
    
Create ZH-tapered isochrone model from configuration.
"""
function create_isochrone_zh_model(config::KalnajsConfig)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity
    
    # ZH taper parameters
    Jc = config.model.Jc
    Rc = config.model.Rc
    eta = config.model.eta
    
    if Jc !== nothing
        return setup_model(IsochroneZHModel, mk; unit_mass, unit_length, selfgravity, Jc, eta)
    elseif Rc !== nothing
        return setup_model(IsochroneZHModel, mk; unit_mass, unit_length, selfgravity, Rc, eta)
    else
        # Default: use R_max from grid configuration
        R_max = 16.0  # Default from grid type 5
        return setup_model(IsochroneZHModel, mk; unit_mass, unit_length, selfgravity, Rc=R_max, eta)
    end
end

"""
    create_isochrone_tanh_model(config::KalnajsConfig) -> ModelResults
    
Create Tanh-tapered isochrone model from configuration.
"""
function create_isochrone_tanh_model(config::KalnajsConfig)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity

    # Energy-dependent tanh taper: Rc not needed
    eta = config.model.eta

    return setup_model(IsochroneTanhModel, mk; unit_mass, unit_length, selfgravity, eta)
end

"""
    create_isochrone_taper_exp_model(config::KalnajsConfig) -> ModelResults
    
Create exponentially tapered isochrone model from configuration.
"""
function create_isochrone_taper_exp_model(config::KalnajsConfig)
    mk = config.model.mk
    unit_mass = config.physics.unit_mass
    unit_length = config.physics.unit_length
    selfgravity = config.physics.selfgravity
    
    eta = config.model.eta
    
    return setup_model(IsochroneTaperExpModel, mk; unit_mass, unit_length, selfgravity, eta)
end

# =============================================================================
# MODEL SETUP FUNCTIONS
# =============================================================================

"""
    setup_model(::Type{IsochroneJHModel}, mk; kwargs...) -> ModelResults
    
Initialize JH-tapered isochrone model.
"""
function setup_model(::Type{IsochroneJHModel}, mk::Int; 
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0)
    
    # Only print if PME_VERBOSE is not "false"
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Isochrone model with JH taper (mk=$mk, selfgravity=$selfgravity)")
    end
    
    # Compute g function via numerical integration
    g_interp = compute_g_function(mk)
    
    # Compute sigma_u function for velocity dispersion
    ppu_interp = compute_sigma_u_function(mk)
    
    # Create core kinematic functions (dimensionless isochrone)
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt( sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0)) 
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))
    
    # Surface density
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)
    
    # Velocity dispersion
    sigma_u(r) = sqrt( ppu_interp(-r.*V(r)) ./ r.^(mk+1) ./ Sigma_d(r) )
   

    # Create distribution functions
    DF, DF_dE, DF_dL = create_jh_distribution_functions(mk, g_interp)
    
    # Package parameters
    params = Dict{String,Any}(
        "mk" => mk, "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "DF_type" => 0, "taper_type" => "JH"
    )
    
    helpers = (
        model_name = "Kalnajs Isochrone - JH Taper",
        title = "JH Isochrone mk=$mk",
        Jr = (E, h) -> 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
    )
    
    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers, "IsochroneJH", params
    )
end

"""
    setup_model(::Type{IsochroneZHModel}, mk; kwargs...) -> ModelResults
    
Initialize ZH-tapered isochrone model.
"""
function setup_model(::Type{IsochroneZHModel}, mk::Int; 
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0,
                     Jc::Union{Real,Nothing}=nothing,
                     Rc::Union{Real,Nothing}=nothing, 
                     eta::Real=1.0)
    
    # Calculate Jc from Rc if needed
    if Jc === nothing && Rc !== nothing
        sc = sqrt(Rc^2 + 1.0)
        Ec = -1.0 / (2.0 * sc)
        Jc = (sc - 1.0) / sqrt(sc) * eta
        # Only print if PME_VERBOSE is not "false"
        if get(ENV, "PME_VERBOSE", "true") != "false"
            println("Setting up Isochrone model with ZH taper (mk=$mk, selfgravity=$selfgravity, Rc=$Rc, Jc=$Jc)")
        end
    elseif Jc !== nothing
        # Only print if PME_VERBOSE is not "false"
        if get(ENV, "PME_VERBOSE", "true") != "false"
            println("Setting up Isochrone model with ZH taper (mk=$mk, selfgravity=$selfgravity, Jc=$Jc)")
        end
    else
        error("Must specify either Jc or Rc for ZH taper")
    end
    
    # Compute g function
    g_interp = compute_g_function(mk)
    
    # Compute sigma_u function for velocity dispersion
    ppu_interp = compute_sigma_u_function(mk)
    
    # Same kinematic functions as JH model
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt( sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0))
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)
    
    # Velocity dispersion
    sigma_u(r) = sqrt(ppu_interp(-V(r) * r) / r^(mk+1) / Sigma_d(r))
    
    # Create ZH distribution functions
    DF, DF_dE, DF_dL = create_zh_distribution_functions(mk, g_interp, Jc)
    
    # Package parameters
    params = Dict{String,Any}(
        "mk" => mk, "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "Jc" => Jc, "eta" => eta, "taper_type" => "ZH"
    )
    if Rc !== nothing
        params["Rc"] = Rc
    end
    
    helpers = (
        model_name = "Kalnajs Isochrone - ZH Taper",
        title = "ZH Isochrone mk=$mk, Jc=$Jc",
        Jr = (E, h) -> 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0,
        taper_function = h -> 0.5 + 3.0*(h/Jc)/4.0 - (h/Jc)^3/4.0
    )
    
    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers, "IsochroneZH", params
    )
end

"""
    setup_model(::Type{IsochroneTanhModel}, mk; kwargs...) -> ModelResults
    
Initialize Tanh-tapered isochrone model.
"""
function setup_model(::Type{IsochroneTanhModel}, mk::Int; 
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0,
                     eta::Real=0.1)
    
    # Only print if PME_VERBOSE is not "false"
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Isochrone model with Tanh taper (energy-dependent Lc) (mk=$mk, selfgravity=$selfgravity, eta=$eta)")
    end
    
    # Compute g function
    g_interp = compute_g_function(mk)
    
    # Compute sigma_u function for velocity dispersion
    ppu_interp = compute_sigma_u_function(mk)
    
    # Same kinematic functions as JH/ZH models
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt( sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0))
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)
    
    # Velocity dispersion
    sigma_u(r) = sqrt(ppu_interp(-V(r) * r) / r^(mk+1) / Sigma_d(r))
    
    # Create energy-dependent tanh DF core
    DF_core, DF_dE_core, DF_dL_core = create_tanh_distribution_functions_energy_dependent(mk, g_interp, eta)
    
    # Wrappers with grid access (OrbitCalculator interface)
    function DF(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing
            error("Grid data is required for Tanh model (energy-dependent Lc)")
        end
        Lc_val = grids.L_m[iR, end]  # circular angular momentum at this radius
        return DF_core(E, h, Lc_val, eta)
    end
    
    function DF_dE(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing
            error("Grid data is required for Tanh model (energy-dependent Lc)")
        end
        Lc_val = grids.L_m[iR, end]
        Rc_val = grids.Rc[iR, 1]
        omega_inv = 1.0 / Omega(Rc_val)  # dLc/dE = 1/Omega at circular orbit
        return DF_dE_core(E, h, Lc_val, omega_inv, eta)
    end
    
    function DF_dL(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing
            error("Grid data is required for Tanh model (energy-dependent Lc)")
        end
        Lc_val = grids.L_m[iR, end]
        return DF_dL_core(E, h, Lc_val, eta)
    end
    
    # Package parameters (Rc removed)
    params = Dict{String,Any}(
        "mk" => mk, "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "eta" => eta, "taper_type" => "TanhEnergy"
    )
    
    helpers = (
        model_name = "Kalnajs Isochrone - Tanh Taper (E-dependent)",
        title = "Tanh Isochrone mk=$mk, eta=$eta",
        Jr = (E, h) -> 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
    )
    
    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers, "IsochroneTanh", params
    )
end
"""
    setup_model(::Type{IsochroneTaperExpModel}, mk; kwargs...) -> ModelResults
    
Initialize exponentially tapered isochrone model.
"""
function setup_model(::Type{IsochroneTaperExpModel}, mk::Int; 
                     unit_mass::Real=1.0, unit_length::Real=1.0, selfgravity::Real=1.0,
                     eta::Real=1.0)
    
    # Only print if PME_VERBOSE is not "false"
    if get(ENV, "PME_VERBOSE", "true") != "false"
        println("Setting up Isochrone model with exponential taper (mk=$mk, selfgravity=$selfgravity, eta=$eta)")
    end
    
    # Compute g function via numerical integration
    g_interp = compute_g_function(mk)
    
    # Compute sigma_u function for velocity dispersion
    ppu_interp = compute_sigma_u_function(mk)
    
    # Create core kinematic functions (dimensionless isochrone)
    V(r) = -1.0 / (1.0 + sqrt(r^2 + 1.0))
    dV(r) = r / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0))^2)
    Omega(r) = 1.0 / sqrt( sqrt(r^2 + 1.0)) / (1.0 + sqrt(r^2 + 1.0)) 
    kappa(r) = sqrt(Omega(r)^2 * (4.0 - r^2 / (r^2 + 1.0) - 2.0 * r^2 / (sqrt(r^2 + 1.0) * (1.0 + sqrt(r^2 + 1.0)))))
    
    # Surface density
    Sigma_d(r) = (log(r + sqrt(r^2 + 1.0)) - r/sqrt(r^2 + 1.0)) / (r^3 * 2π)
    
    # Velocity dispersion
    sigma_u(r) = sqrt( ppu_interp(-r.*V(r)) ./ r.^(mk+1) ./ Sigma_d(r) )

    # Basic taper functions (functional form)
    taper(x) = 1.0 - exp(-x^2)
    taper_deriv(x) = -2.0 * x * exp(-x^2)
    
    # Circular orbit angular momentum function for isochrone potential
    # Lc(Rc) = (s-1)/sqrt(s) where s = sqrt(Rc^2 + 1)
    function Lc(Rc::Real)
        s = sqrt(Rc^2 + 1.0)
        return (s - 1.0) / sqrt(s)
    end
    
    # Derivative dLc/dE = 1/Omega(Rc) for chain rule
    function dLc_dE(Rc::Real)
        return 1.0 / Omega(Rc)
    end

    # Create distribution functions that can be called directly with grid access
    DF_core, DF_dE_core, DF_dL_core = create_exp_tapered_distribution_functions_simple(mk, g_interp, eta, taper, taper_deriv, Lc, dLc_dE)
    
    # Create wrapper functions that match the expected OrbitCalculator interface
    function DF(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing
            error("Grid data is required for exponential taper model but not provided")
        end
        Lc_val = grids.L_m[iR, end]  # Circular orbit L at radius iR
        return DF_core(E, h, Lc_val, eta)
    end
    
    function DF_dE(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing
            error("Grid data is required for exponential taper model but not provided")
        end
        Lc_val = grids.L_m[iR, end]  # Circular orbit L at radius iR
        Rc_val = grids.Rc[iR, 1]    # Circular radius at radius index iR
        omega_inv = 1.0 / Omega(Rc_val)  # 1/Omega_2 at circular orbit
        return DF_dE_core(E, h, Lc_val, omega_inv, eta)
    end
    
    function DF_dL(E::Real, h::Real, iR::Int=-1, iv::Int=-1, grids=nothing)::Real
        if grids === nothing
            error("Grid data is required for exponential taper model but not provided")
        end
        Lc_val = grids.L_m[iR, end]  # Circular orbit L at radius iR
        return DF_dL_core(E, h, Lc_val, eta)
    end
    
    # Package parameters
    params = Dict{String,Any}(
        "mk" => mk, "unit_mass" => unit_mass, "unit_length" => unit_length,
        "selfgravity" => selfgravity, "eta" => eta, "taper_type" => "Exp"
    )
    
    helpers = (
        model_name = "Kalnajs Isochrone - Exp Taper",
        title = "Exp Tapered Isochrone mk=$mk, eta=$eta",
        Jr = (E, h) -> 1.0/sqrt(-2.0*E) - (h + sqrt(h^2 + 4.0))/2.0,
        taper_function = (L, Rc) -> taper(L / (eta * Lc(Rc)))
    )
    
    return ModelResults{Float64}(
        V, dV, Omega, kappa,
        DF, DF_dE, DF_dL,
        Sigma_d, sigma_u,
        helpers, "IsochroneTaperExp", params
    )
end

# =============================================================================
# G FUNCTION COMPUTATION
# =============================================================================

"""
    compute_g_function(m::Int) -> Interpolations.AbstractInterpolation

Compute the g(x) function for isochrone model via numerical integration.
Returns an interpolating function for efficient evaluation.
"""
function compute_g_function(m::Int)
    # Computing g(x) function silently
    
    # Grid setup (optimized spacing)
    x_eps = 1e-5
    x = reverse(1.0 .- 10.0 .^ range(log10(x_eps), 0, length=1001))
    # x = collect(range(0.0, 1.0-x_eps, length=1001))
    
    # Coordinate transformation
    r = 2.0 * x ./ (1.0 .- x.^2)
    d_r = 2.0 * (x.^2 .+ 1.0) ./ (1.0 .- x.^2).^2
    
    # Isochrone helper functions
    s = sqrt.(r.^2 .+ 1.0)
    L = log.(r .+ s) .- r ./ s
    tau = L .* (s .+ 1.0).^m ./ (2π) ./ r.^3
    
    # Derivatives
    d_L_dr = r.^2 ./ s.^3
    d_s_dr = r ./ s
    d_tau_dr = tau .* (d_L_dr ./ L .+ m ./ (1.0 .+ s) .* r ./ s .- 3.0 ./ r)
    
    # Components
    g1 = m * tau .+ x .* d_tau_dr .* d_r
    g2 = m * (m - 1) / 2.0 * tau
    
    # Compute g3 via numerical integration, starting from index 2
    P_coeffs, coeffs_desc = get_legendre_derivatives(m - 1)
    
    g3 = zeros(length(x))   
    # Start from index 2 to avoid singularities
    for i in 2:length(x)
        xi = x[i]
        integrand(t) = integrand_function(t, xi, m, coeffs_desc)
        try
            g3[i], _ = quadgk(integrand, 0.0, 1.0, rtol=1e-10, atol=1e-12)
        catch
            g3[i] = 0.0
        end        
    end
    
    # Final result
    g_total = (g1 .- g2 .+ g3) ./ π
    
    # Set boundary value analytically: g_total(1) = m/(3π²)
    g_total[1] = m / (3 * π^2)
    
    # Create interpolation
    # Using high precision interpolation
    g_spline = cubic_spline_interp(x, g_total)
    
    # Evaluate spline and its derivative on regular grid for verification
    # x_regular = collect(range(0.0, 0.999, length=1001))
    # g_values = [g_spline(xi) for xi in x_regular]
    # g_derivatives = [derivative(g_spline, xi) for xi in x_regular]
    
    return g_spline
end

"""
Compute integrand for g3 calculation.
"""
function integrand_function(t::Real, xi::Real, m::Int, coeffs_desc::Vector{Float64})

    if t >= 1.0
        return 0.0
    end
    
    u = t * xi
    if abs(u) >= 1.0 - 1e-12
        return 0.0
    end
    
    # Calculate r = 2u/(1-u²)
    r_val = 2.0 * u / (1.0 - u^2)
    
    # Calculate tau(r)
    s_val = sqrt(r_val^2 + 1.0)
    L_val = log(r_val + s_val) - r_val / s_val
    
    if r_val >= 1e-3
        tau_val = L_val * (s_val + 1.0)^m / (2π) / r_val^3 
    else
        # Taylor expansion for small r
        tau_val = (-35*r_val^6/144 + 15*r_val^4/56 - 3*r_val^2/10 + 1/3) / 2π * 2^m
    end
    
    # Legendre polynomial second derivative
    P_double_prime_val = evalpoly(t, coeffs_desc)
    
    return tau_val * t^m * P_double_prime_val
end

"""
    get_legendre_derivatives(n::Int) -> (Vector{Float64}, Vector{Float64})

Compute Legendre polynomial coefficients and second derivative coefficients.
"""
function get_legendre_derivatives(n::Int)
    if n == 0
        return [1.0], [0.0]
    elseif n == 1
        return [0.0, 1.0], [0.0]
    end
    
    # Recurrence relation for Legendre polynomials
    P0 = [1.0]
    P1 = [0.0, 1.0]
    
    if n == 1
        P_coeffs = P1
    else
        for k in 1:(n-1)
            # (k+1)*P_{k+1}(x) = (2k+1)*x*P_k(x) - k*P_{k-1}(x)
            xPk = vcat([0.0], P1)
            target_length = k + 2
            
            xPk_padded = vcat(xPk, zeros(target_length - length(xPk)))
            P0_padded = vcat(P0, zeros(target_length - length(P0)))
            
            P_new = ((2*k + 1) * xPk_padded - k * P0_padded) / (k + 1)
            
            P0 = P1
            P1 = P_new[1:target_length]
        end
        P_coeffs = P1
    end
    
    # Compute second derivative
    P_double_prime_coeffs = compute_second_derivative_coeffs(P_coeffs)
    
    return P_coeffs, P_double_prime_coeffs
end

"""
    compute_second_derivative_coeffs(poly_coeffs::Vector{Float64}) -> Vector{Float64}

Compute second derivative coefficients from polynomial coefficients.
"""
function compute_second_derivative_coeffs(poly_coeffs::Vector{Float64})
    if length(poly_coeffs) < 3
        return [0.0]
    end
    
    n = length(poly_coeffs)
    second_deriv_coeffs = zeros(max(1, n - 2))
    
    for k in 3:n
        power = k - 1
        coeff = poly_coeffs[k]
        new_power = power - 2
        new_coeff = coeff * power * (power - 1)
        
        if new_power >= 0 && new_power + 1 <= length(second_deriv_coeffs)
            second_deriv_coeffs[new_power + 1] = new_coeff
        end
    end
    
    # Remove trailing zeros
    while length(second_deriv_coeffs) > 1 && abs(second_deriv_coeffs[end]) < 1e-15
        pop!(second_deriv_coeffs)
    end
    
    if all(abs.(second_deriv_coeffs) .< 1e-15)
        return [0.0]
    end
    
    return second_deriv_coeffs
end

# =============================================================================
# SIGMA_U FUNCTION COMPUTATION
# =============================================================================

"""
    compute_sigma_u_function(m::Int) -> Interpolations.AbstractInterpolation

Compute sigma_u(r) function for velocity dispersion via numerical integration.
Returns an interpolating function for efficient evaluation.
"""
function compute_sigma_u_function(m::Int)
    # Computing sigma_u function silently

    # Grid setup - EXACT MATLAB MATCH
    # MATLAB: xu_esp = 1e-6; xu = 1 - logspace(log10(xu_esp), 0, 10001); xu = fliplr(xu);
    xu_eps = 1e-6
    logspace_values = 10.0 .^ range(log10(xu_eps), 0, length=10001)
    xu_temp = 1.0 .- logspace_values
    xu = reverse(xu_temp)  # fliplr in MATLAB
    dxu = diff(xu)

    # Calculate tau_f for each xu value - EXACT MATLAB MATCH
    # MATLAB: t_tau = tau_f(xu,m).*(2*xu).^m;
    t_tau = [tau_f(u, m) for u in xu] .* (2.0 .* xu).^m

    # Numerical integration using trapezoidal rule - EXACT MATLAB MATCH
    # MATLAB: gu(2:end) = (t_tau(1:end-1) + t_tau(2:end)) .* dxu/2;
    gu = zeros(length(xu))
    gu[2:end] .= (t_tau[1:end-1] .+ t_tau[2:end]) .* dxu ./ 2.0

    # Cumulative sum for the integral - EXACT MATLAB MATCH
    # MATLAB: ppu = csaps(xu, cumsum(gu), 1);
    cumulative_gu = cumsum(gu)

    
    # Using high precision interpolation for sigma_u
    sigma_u_spline = cubic_spline_interp(xu, cumulative_gu)

    return sigma_u_spline
end

"""
    tau_f(u, m) -> Float64

Calculate tau function used for sigma_u calculation.
"""
function tau_f(u::Real, m::Int)
    if abs(u) >= 1.0 - 1e-12
        return 0.0
    end
    
    # Calculate r = 2u/(1-u²)
    r_val = 2.0 * u / (1.0 - u^2)
    
    # Calculate tau(r)
    s_val = sqrt(r_val^2 + 1.0)
    L_val = log(r_val + s_val) - r_val / s_val
    
    if r_val >= 1e-3
        tau_val = L_val * ((s_val + 1.0)/2.0)^m / (2π) / r_val^3 
    else
        # Taylor expansion for small r
        tau_val = (-35.0*r_val^6/144.0 + 15.0*r_val^4/56.0 - 3.0*r_val^2/10.0 + 1/3.0) / 2π 
    end

    return tau_val
end


# =============================================================================

"""
    create_jh_distribution_functions(m, g_interp) -> (Function, Function, Function)

Create JH-tapered distribution function and its derivatives.
"""
# =============================================================================
# PLAIN DISTRIBUTION FUNCTIONS (NO TAPER)
# =============================================================================

"""
    create_plain_distribution_functions(m, g_interp) -> (Function, Function, Function)

Create plain (untapered) distribution function and its derivatives.
Unidirectional formulation (L ≥ 0 only).
DF uses Kalnajs g-function which already includes proper normalization.
DF is even in L: f(E,L) = f(E,-L).
"""
function create_plain_distribution_functions(m::Int, g_interp)
    
    # Distribution function - unidirectional (L ≥ 0)
    # Normalization: 2/(2π)² to match Miyamoto convention
    function DF(E::Real, h::Real)::Real
        if E >= 0.0 || h < 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * h
        main_term = (-2.0 * E)^(m - 1) * g_interp(xi)
        
        # Kalnajs g-function already includes proper normalization
        # No additional factor needed for plain isochrone
        return main_term
    end
    
    # E-derivative
    function DF_dE(E::Real, h::Real)::Real
        if E >= 0.0 || h < 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * h
        g_val = g_interp(xi)
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end
        
        term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
        term2 = (-2.0*E)^(m - 1) * g_prime * (-h / sqrt(-2.0*E))
        
        return term1 + term2
    end
    
    # L-derivative
    function DF_dL(E::Real, h::Real)::Real
        if E >= 0.0 || h < 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * h
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end
        
        term = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E)
        
        return term
    end
    
    return DF, DF_dE, DF_dL
end


function create_jh_distribution_functions(m::Int, g_interp)
    
    # Distribution function - EXACT MATLAB MATCH
    # MATLAB: DF = @(E,h) ( (-2*E).^(m-1) .* fnval(pp, sqrt(-2*E).*h) - ...
    #             m/(6*pi^2) ./ (1 + Jr(E,h) + abs(h)).^(2*m-2) ).* (h > 0) + ...
    #             m/(6*pi^2) ./ (1 + Jr(E,h) + abs(h)).^(2*m-2) .* (h <= 0);
    function DF(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        # Calculate Jr exactly as in MATLAB: Jr = 1/sqrt(-2*E) - (|h| + sqrt(h^2 + 4))/2
        Jr = 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
        taper_term = m / (6.0*π^2) / (1.0 + Jr + abs(h))^(2*m - 2)
        
        if h >= 0.0
            # Both terms: main term - taper term
            xi = sqrt(-2.0 * E) * h
            g_val = g_interp(xi)
            main_term = (-2.0 * E)^(m - 1) * g_val
            return main_term - taper_term
        else
            # h <= 0: only taper term
            return taper_term
        end
    end
    
    # E-derivative - EXACT MATLAB MATCH (derivative_JH_DF_E) with analytical derivatives
    function DF_dE(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        Jr = 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
        dJr_dE = 1.0 / ((-2.0*E)^(3/2))
        
        if h > 0.0
            xi = sqrt(-2.0 * E) * h
            g_val = g_interp(xi)
            
            # Compute derivative of g_interp using analytical spline derivative
            g_prime = try
                derivative(g_interp, xi)  # Analytical derivative using Dierckx
            catch
                0.0
            end
            
            # MATLAB terms: term1a + term1b + term2
            term1a = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
            term1b = (-2.0*E)^(m - 1) * g_prime * (-h / sqrt(-2.0*E))
            term2 = m/(6.0*π^2) * (2*m - 2) * dJr_dE / (1.0 + Jr + abs(h))^(2*m - 1)
            
            return term1a + term1b + term2
        elseif h < 0.0  # Note: MATLAB uses h < 0, not h <= 0
            # Only taper term derivative (negative)
            return -m/(6.0*π^2) * (2*m - 2) * dJr_dE / (1.0 + Jr + abs(h))^(2*m - 1)
        else
            # h = 0: Use right-hand limit (h -> 0+)
            xi = 0.0
            g_val = g_interp(xi)
            
            # term1b = 0 since h = 0
            term1a = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
            # Jr at h=0: Jr = 1/sqrt(-2*E) - 2/2 = 1/sqrt(-2*E) - 1
            Jr_zero = 1.0/sqrt(-2.0*E) - 1.0
            term2 = m/(6.0*π^2) * (2*m - 2) * dJr_dE / (1.0 + Jr_zero)^(2*m - 1)
            
            return term1a + term2
        end
    end
    
    # L-derivative - EXACT MATLAB MATCH (derivative_JH_DF_h)
    function DF_dL(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        Jr = 1.0/sqrt(-2.0*E) - (abs(h) + sqrt(h^2 + 4.0))/2.0
        
        # dJr/dh and d|h|/dh - EXACT MATLAB MATCH
        # MATLAB: dJr_dh_pos = -(1 + h_pos./sqrt(h_pos.^2 + 4))/2
        # MATLAB: dJr_dh = -(-1 + h_neg./sqrt(h_neg.^2 + 4))/2 (for h<0)
        if h > 0.0
            dJr_dh = -(1.0 + h/sqrt(h^2 + 4.0))/2.0
            dabs_h_dh = 1.0  # d|h|/dh = 1 for h > 0
        else
            dJr_dh = -(-1.0 + h/sqrt(h^2 + 4.0))/2.0
            dabs_h_dh = -1.0  # d|h|/dh = -1 for h < 0
        end
        
        if h > 0.0
            # Positive h case - MATLAB: term1 + term2
            xi = sqrt(-2.0 * E) * h
            
            # Compute derivative of g_interp using analytical spline derivative
            g_prime = try
                derivative(g_interp, xi)  # Analytical derivative using Dierckx
            catch
                0.0
            end
            
            # MATLAB logic: term1 + term2
            term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E)
            term2 = m/(6.0*π^2) * (2*m - 2) * (dJr_dh + dabs_h_dh) / (1.0 + Jr + abs(h))^(2*m - 1)
            return term1 + term2
        elseif h < 0.0
            # Negative h case - MATLAB: -term2
            term2 = m/(6.0*π^2) * (2*m - 2) * (dJr_dh + dabs_h_dh) / (1.0 + Jr + abs(h))^(2*m - 1)
            return -term2
        else
            # h = 0 case - Use right limit (h → 0+)
            g_prime = try
                derivative(g_interp, 0.0)  # Analytical derivative at zero
            catch
                0.0
            end
            
            term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E)
            # dJr_dh = -1/2, dabs_h_dh = 1 at right limit
            term2 = m/(6.0*π^2) * (2*m - 2) * 0.5 / (1.0 + Jr)^(2*m - 1)
            return term1 + term2
        end
    end
    
    return DF, DF_dE, DF_dL
end

"""
    create_zh_distribution_functions(m, g_interp, Jc) -> (Function, Function, Function)

Create ZH-tapered distribution function and its derivatives.
"""
function create_zh_distribution_functions(m::Int, g_interp, Jc::Real)
    
    # Taper function
    taper(h) = 0.5 + 3.0*(h/Jc)/4.0 - (h/Jc)^3/4.0
    taper_deriv(h) = (3.0/4.0/Jc) - (3.0*h^2)/(4.0*Jc^3)
    
    # Distribution function (original Kalnajs formulation)
    function DF(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        if h > Jc
            xi = sqrt(-2.0 * E) * h
            return (-2.0 * E)^(m - 1) * g_interp(xi)
        else  # abs(h) <= Jc
            xi = sqrt(-2.0 * E) * abs(h)
            return (-2.0 * E)^(m - 1) * g_interp(xi) * taper(h)
        end
    end
    
    # E-derivative with analytical spline derivatives
    function DF_dE(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        if h > Jc
            xi = sqrt(-2.0 * E) * h
            g_val = g_interp(xi)
            
            # Compute derivative of g_interp using analytical spline derivative
            g_prime = try
                derivative(g_interp, xi)  # Analytical derivative using Dierckx
            catch
                0.0
            end
            
            term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
            term2 = (-2.0*E)^(m - 1) * g_prime * (-h / sqrt(-2.0*E))
            return term1 + term2
        else  # abs(h) <= Jc
            xi = sqrt(-2.0 * E) * abs(h)
            g_val = g_interp(xi)
            taper_val = taper(h)
            
            # Compute derivative of g_interp using analytical spline derivative
            g_prime = try
                derivative(g_interp, xi)  # Analytical derivative using Dierckx
            catch
                0.0
            end
            
            term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
            term2 = (-2.0*E)^(m - 1) * g_prime * (-abs(h) / sqrt(-2.0*E))
            return (term1 + term2) * taper_val
        end
    end
    
    # L-derivative with analytical spline derivatives
    function DF_dL(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        if h > Jc
            xi = sqrt(-2.0 * E) * h
            
            # Compute derivative of g_interp using analytical spline derivative
            g_prime = try
                derivative(g_interp, xi)  # Analytical derivative using Dierckx
            catch
                0.0
            end
            
            return (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E)
        else  # abs(h) <= Jc
            xi = sqrt(-2.0 * E) * abs(h)
            g_val = g_interp(xi)
            taper_val = taper(h)
            taper_deriv_val = taper_deriv(h)
            
            # Compute derivative of g_interp using analytical spline derivative
            g_prime = try
                derivative(g_interp, xi)  # Analytical derivative using Dierckx
            catch
                0.0
            end
            
            term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E) * sign(h) * taper_val
            term2 = (-2.0*E)^(m - 1) * g_val * taper_deriv_val
            return term1 + term2
        end
    end
    
    return DF, DF_dE, DF_dL
end

"""
    create_tanh_distribution_functions(m, g_interp, Lc, eta) -> (Function, Function, Function)

Create Tanh-tapered distribution function and its derivatives.
"""
function create_tanh_distribution_functions(m::Int, g_interp, Lc::Real, eta::Real)
    
    # Tanh taper function and its derivative
    taper(h) = (1.0 + tanh(h / (Lc * eta))) / 2.0
    taper_deriv(h) = (1.0 / (2.0 * Lc * eta)) * (1.0 - tanh(h / (Lc * eta))^2)
    
    # Distribution function (Kalnajs formulation with tanh taper)
    function DF(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * abs(h)
        main_term = (-2.0 * E)^(m - 1) * g_interp(xi)
        taper_val = taper(h)
        
        return main_term * taper_val
    end
    
    # E-derivative with analytical spline derivatives
    function DF_dE(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        taper_val = taper(h)
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)  # Analytical derivative using Dierckx
        catch
            0.0
        end
        
        term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
        term2 = (-2.0*E)^(m - 1) * g_prime * (-abs(h) / sqrt(-2.0*E))
        
        return (term1 + term2) * taper_val
    end
    
    # L-derivative with analytical spline derivatives
    function DF_dL(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        taper_val = taper(h)
        taper_deriv_val = taper_deriv(h)
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)  # Analytical derivative using Dierckx
        catch
            0.0
        end
        
        term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E) * sign(h) * taper_val
        term2 = (-2.0*E)^(m - 1) * g_val * taper_deriv_val
        
        return term1 + term2
    end
    
    return DF, DF_dE, DF_dL
end

"""
    create_exp_tapered_distribution_functions(m, g_interp, L0, eta, taper, taper_deriv) -> (Function, Function, Function)

Create exponentially tapered distribution function and its derivatives.
"""
function create_exp_tapered_distribution_functions(m::Int, g_interp, L0::Real, eta::Real, taper::Function, taper_deriv::Function)
    
    # Distribution function with exponential taper similar to ExpDisk model
    function DF(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * abs(h)
        main_term = (-2.0 * E)^(m - 1) * g_interp(xi)
        taper_val = taper(abs(h))  # Apply taper to |h| like ExpDisk L0 parameter
        
        return main_term * taper_val
    end
    
    # E-derivative with analytical spline derivatives
    function DF_dE(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        taper_val = taper(abs(h))
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)  # Analytical derivative using Dierckx
        catch
            0.0
        end
        
        term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
        term2 = (-2.0*E)^(m - 1) * g_prime * (-abs(h) / sqrt(-2.0*E))
        
        return (term1 + term2) * taper_val
    end
    
    # L-derivative with analytical spline derivatives
    function DF_dL(E::Real, h::Real)::Real
        if E >= 0.0
            return 0.0
        end
        
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        taper_val = taper(abs(h))
        taper_deriv_val = taper_deriv(abs(h)) * sign(h)  # Chain rule for d/dh taper(|h|)
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)  # Analytical derivative using Dierckx
        catch
            0.0
        end
        
        term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E) * sign(h) * taper_val
        term2 = (-2.0*E)^(m - 1) * g_val * taper_deriv_val
        
        return term1 + term2
    end
    
    return DF, DF_dE, DF_dL
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    create_exp_tapered_distribution_functions_simple(m, g_interp, eta, taper, taper_deriv, Lc, dLc_dE) -3e (Function, Function, Function)

Simplified exponentially tapered distribution function and its derivatives with grid access.
"""
function create_exp_tapered_distribution_functions_simple(m::Int, g_interp, eta::Real, taper::Function, taper_deriv::Function, Lc::Function, dLc_dE::Function)
    
function DF(E::Real, h::Real, Lc_val::Real, eta::Real)::Real
    if E >= 0.0
        return 0.0
    end
    
    xi = sqrt(-2.0 * E) * h
    main_term = (-2.0 * E)^(m - 1) * g_interp(xi)
    taper_val = taper(h / (eta * Lc_val))
    
    return main_term * taper_val
end
    
function DF_dE(E::Real, h::Real, Lc_val::Real, omega_inv::Real, eta::Real)::Real
    if E >= 0.0
        return 0.0
    end
    
    xi = sqrt(-2.0 * E) * h
    g_val = g_interp(xi)
    g_prime = derivative(g_interp, xi)

    term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
    term2 = (-2.0*E)^(m - 1) * g_prime * (-h / sqrt(-2.0*E))
    
    x_arg = h / (eta * Lc_val)
    taper_val = taper(x_arg)
    taper_deriv_val = taper_deriv(x_arg)
    dtaper_dE = taper_deriv_val * (-x_arg/Lc_val) * omega_inv
    
    return (term1 + term2) * taper_val + (-2.0*E)^(m - 1) * g_val * dtaper_dE
end
    
function DF_dL(E::Real, h::Real, Lc_val::Real, eta::Real)::Real
    if E >= 0.0
        return 0.0
    end
    
    xi = sqrt(-2.0 * E) * h
    g_val = g_interp(xi)
    g_prime = derivative(g_interp, xi)
    
    x_arg = h / (eta * Lc_val)
    taper_val = taper(x_arg)
    taper_deriv_val = taper_deriv(x_arg)
    dtaper_dL = taper_deriv_val / (eta * Lc_val)
    
    term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E) * taper_val
    term2 = (-2.0*E)^(m - 1) * g_val * dtaper_dL
    
    return term1 + term2
end
    
    return DF, DF_dE, DF_dL
end

"""
    create_exp_tapered_distribution_functions_grid(m, g_interp, eta, taper, taper_deriv, Lc, dLc_dE, grids) -> (Function, Function, Function)

Create exponentially tapered distribution function and its derivatives using grid-based approach.
"""
function create_exp_tapered_distribution_functions_grid(m::Int, g_interp, eta::Real, taper::Function, taper_deriv::Function, Lc::Function, dLc_dE::Function, grids)
    
    # Get grid dimensions
    NR, Nv = size(grids.L_m)
    
    # Distribution function with exponential taper using grid-based Lc lookup
    function DF(E::Real, h::Real, iR::Int, iv::Int)::Real
        if E >= 0.0
            return 0.0
        end
        
        # Use circular orbit angular momentum: Lc = L_m[iR, Nv]
        Lc_val = grids.L_m[iR, Nv]
        
        xi = sqrt(-2.0 * E) * h  # Removed abs(h) since L >= 0 always
        main_term = (-2.0 * E)^(m - 1) * g_interp(xi)
        taper_val = taper(h / (eta * Lc_val))
        
        return main_term * taper_val
    end
    
    # E-derivative with chain rule for energy-dependent Lc
    function DF_dE(E::Real, h::Real, iR::Int, iv::Int)::Real
        if E >= 0.0
            return 0.0
        end
        
        # Use circular orbit angular momentum: Lc = L_m[iR, Nv]
        Lc_val = grids.L_m[iR, Nv]
        
        # For chain rule, we need dLc/dE = 1/Omega_1 at circular orbit
        # Use Rc[iR, 1] since all Rc values at same iR are the same
        Rc_val = grids.Rc[iR, 1]
        dLc_dE_val = dLc_dE(Rc_val)
        
        xi = sqrt(-2.0 * E) * h  # Removed abs(h)
        g_val = g_interp(xi)
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)  # Analytical derivative using Dierckx
        catch
            0.0
        end
        
        # Standard terms from original DF
        term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
        term2 = (-2.0*E)^(m - 1) * g_prime * (-h / sqrt(-2.0*E))
        
        # Taper function and its derivative
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        taper_deriv_val = taper_deriv(x_arg)
        
        # Chain rule for taper: d/dE taper(h/(eta*Lc(E))) = taper'(x) * (-h/(eta*Lc^2)) * dLc/dE
        dtaper_dE = taper_deriv_val * (-h / (eta * Lc_val^2)) * dLc_dE_val
        
        # User's formula: ∂F/∂E = H*∂f/∂E + f(E,L) * H' * (-L/Lc^2) / Omega_1
        # where H' = dtaper/dE and f is the main_term
        return (term1 + term2) * taper_val + (-2.0*E)^(m - 1) * g_val * dtaper_dE
    end
    
    # L-derivative with chain rule
    function DF_dL(E::Real, h::Real, iR::Int, iv::Int)::Real
        if E >= 0.0
            return 0.0
        end
        
        # Use circular orbit angular momentum: Lc = L_m[iR, Nv]
        Lc_val = grids.L_m[iR, Nv]
        
        xi = sqrt(-2.0 * E) * h  # Removed abs(h)
        g_val = g_interp(xi)
        
        # Compute derivative of g_interp using analytical spline derivative
        g_prime = try
            derivative(g_interp, xi)  # Analytical derivative using Dierckx
        catch
            0.0
        end
        
        # Taper function and its derivative
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        taper_deriv_val = taper_deriv(x_arg)
        
        # Chain rule for taper: d/dL taper(L/(eta*Lc)) = taper'(x) * (1/(eta*Lc))
        dtaper_dL = taper_deriv_val / (eta * Lc_val)
        
        term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E) * taper_val  # Removed sign(h)
        term2 = (-2.0*E)^(m - 1) * g_val * dtaper_dL
        
        return term1 + term2
    end
    
    return DF, DF_dE, DF_dL
end

"""
    cubic_spline_interp(x, y) -> Dierckx.Spline1D

Create cubic spline interpolation using Dierckx with analytical derivatives.
"""
function cubic_spline_interp(x::Vector{T}, y::Vector{T}) where T
    # Use Dierckx for cubic spline with analytical derivatives
    return Spline1D(x, y, k=3)  # k=3 for cubic spline
end

"""
    create_tanh_distribution_functions_energy_dependent(m, g_interp, eta) -> (DF_core, DF_dE_core, DF_dL_core)

Create tanh-tapered DF and derivatives with energy-dependent Lc(E) via wrappers.
"""
function create_tanh_distribution_functions_energy_dependent(m::Int, g_interp, eta::Real)
    taper(x) = (1.0 + tanh(x)) / 2.0
    taper_deriv(x) = 0.5 * (1.0 - tanh(x)^2)

    function DF_core(E::Real, h::Real, Lc_val::Real, eta::Real)::Real
        if E >= 0.0
            return 0.0
        end
        xi = sqrt(-2.0 * E) * abs(h)
        main_term = (-2.0 * E)^(m - 1) * g_interp(xi)
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        return main_term * taper_val
    end

    function DF_dE_core(E::Real, h::Real, Lc_val::Real, omega_inv::Real, eta::Real)::Real
        if E >= 0.0
            return 0.0
        end
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end
        term1 = (m - 1) * (-2.0*E)^(m - 2) * (-2.0) * g_val
        term2 = (-2.0*E)^(m - 1) * g_prime * (-abs(h) / sqrt(-2.0*E))
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        taper_d = taper_deriv(x_arg)
        dtaper_dE = taper_d * (-h / (eta * Lc_val^2)) * omega_inv
        return (term1 + term2) * taper_val + (-2.0*E)^(m - 1) * g_val * dtaper_dE
    end

    function DF_dL_core(E::Real, h::Real, Lc_val::Real, eta::Real)::Real
        if E >= 0.0
            return 0.0
        end
        xi = sqrt(-2.0 * E) * abs(h)
        g_val = g_interp(xi)
        g_prime = try
            derivative(g_interp, xi)
        catch
            0.0
        end
        x_arg = h / (eta * Lc_val)
        taper_val = taper(x_arg)
        taper_d = taper_deriv(x_arg)
        dtaper_dL = taper_d / (eta * Lc_val)
        term1 = (-2.0*E)^(m - 1) * g_prime * sqrt(-2.0*E) * sign(h) * taper_val
        term2 = (-2.0*E)^(m - 1) * g_val * dtaper_dL
        return term1 + term2
    end

    return DF_core, DF_dE_core, DF_dL_core
end


end # module Isochrone


