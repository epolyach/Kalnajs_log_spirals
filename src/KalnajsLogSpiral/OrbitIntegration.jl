# src/KalnajsLogSpiral/OrbitIntegration.jl
"""
Orbit integration for Kalnajs log-spiral eigenvalue problem.

Computes orbital frequencies (Omega_1, Omega_2) and orbital trajectories
for use in the matrix element calculations.

Based on PME's OrbitCalculator.jl with corrections for the Toomre model.
"""
module OrbitIntegration

using LinearAlgebra
using ..Configuration
using ..Models

export OrbitData, compute_grids, compute_orbits
export GridData

# ============================================================================
# Grid Data Structure
# ============================================================================

"""
Grid data for phase space sampling
"""
struct GridData{T<:AbstractFloat}
    Rc::Matrix{T}      # Circular radius grid [NR, Ne]
    e::Matrix{T}       # Eccentricity [NR, Ne]
    v::Matrix{T}       # Velocity parameter (e = 1 - |v|) [NR, Ne]
    R1::Matrix{T}      # Pericenter [NR, Ne]
    R2::Matrix{T}      # Apocenter [NR, Ne]
    E::Matrix{T}       # Energy [NR, Ne]
    L_m::Matrix{T}     # Angular momentum [NR, Ne]
    L2_m::Matrix{T}    # Angular momentum squared [NR, Ne]
    SGNL::Matrix{T}    # Sign of L [NR, Ne]
    S_RC::Vector{T}    # Radial integration weights
    S_e::Matrix{T}     # Eccentricity integration weights
end

"""
Orbit data including frequencies and trajectories
"""
struct OrbitData{T<:AbstractFloat}
    grids::GridData{T}
    Omega_1::Matrix{T}     # Radial frequency [NR, Ne]
    Omega_2::Matrix{T}     # Azimuthal frequency [NR, Ne]
    w1::Array{T,3}         # Radial phase [nwa, NR, Ne]
    ra::Array{T,3}         # Radial position [nwa, NR, Ne]
    pha::Array{T,3}        # Azimuthal phase [nwa, NR, Ne]
    jacobian::Matrix{T}    # Jacobian [NR, Ne]
    F0::Matrix{T}          # Distribution function [NR, Ne]
    FE::Matrix{T}          # ∂f/∂E [NR, Ne]
    FL::Matrix{T}          # ∂f/∂L [NR, Ne]
end

# ============================================================================
# Grid Construction
# ============================================================================

"""
Trapezoidal integration coefficients
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
Simpson integration coefficients
"""
function simpson_coef(x::AbstractVector{T}) where T
    n = length(x)
    if n < 3 || mod(n, 2) == 0
        return trapezoid_coef(x)
    end
    dx = x[2] - x[1]
    S = zeros(T, n)
    S[1] = dx / 3
    S[end] = dx / 3
    for i in 2:2:n-1
        S[i] = 4 * dx / 3
    end
    for i in 3:2:n-2
        S[i] = 2 * dx / 3
    end
    return S
end

"""
    compute_grids(config, model) -> GridData

Compute phase space grids (Rc, e) and derived quantities (E, L).
"""
function compute_grids(config::KalnajsConfig, model)
    T = Float64
    NR = config.grid.NR
    Ne = config.grid.Ne
    Rc_min = T(config.grid.Rc_min)
    Rc_max = T(config.grid.Rc_max)
    v_min = T(config.grid.v_min)
    
    # Logarithmic Rc grid
    Rc_1d = T.(10 .^ range(log10(Rc_min), log10(Rc_max), length=NR))
    uRc = log.(Rc_1d)
    S_RC = Rc_1d .* trapezoid_coef(uRc)
    
    # Eccentricity grid for each Rc
    Rc = zeros(T, NR, Ne)
    e = zeros(T, NR, Ne)
    v = zeros(T, NR, Ne)
    S_e = zeros(T, NR, Ne)
    
    for iR in 1:NR
        v_arr = T.(range(v_min, T(1), length=Ne))
        S_e[iR, :] .= simpson_coef(v_arr)
        v[iR, :] .= v_arr
        Rc[iR, :] .= Rc_1d[iR]
    end
    
    # Small v -> 1e-12 to avoid division by zero
    v[abs.(v) .< T(1e-14)] .= T(1e-12)
    SGNL = sign.(v)
    
    e .= T(1) .- abs.(v)
    R1 = Rc .* (T(1) .- e)
    R2 = Rc .* (T(1) .+ e)
    
    # Energy from turning points
    E = zeros(T, NR, Ne)
    for iR in 1:NR
        for ie in 1:Ne
            r1, r2 = R1[iR, ie], R2[iR, ie]
            V1, V2 = Models.potential(model, r1), Models.potential(model, r2)
            
            if abs(r2^2 - r1^2) > T(1e-14)
                E[iR, ie] = (V2 * r2^2 - V1 * r1^2) / (r2^2 - r1^2)
            else
                # Circular limit
                r = Rc[iR, ie]
                E[iR, ie] = r * Models.potential_derivative(model, r) / 2 + Models.potential(model, r)
            end
            
            if !isfinite(E[iR, ie])
                r = R2[iR, ie]
                E[iR, ie] = Models.potential(model, r)
            end
        end
    end
    
    # Angular momentum squared
    L2_m = zeros(T, NR, Ne)
    for iR in 1:NR
        for ie in 1:Ne
            r2 = R2[iR, ie]
            L2_m[iR, ie] = T(2) * (E[iR, ie] - Models.potential(model, r2)) * r2^2
        end
    end
    L2_m[L2_m .< T(1e-14)] .= zero(T)
    L_m = sqrt.(L2_m) .* SGNL
    
    return GridData{T}(Rc, e, v, R1, R2, E, L_m, L2_m, SGNL, S_RC, S_e)
end

# ============================================================================
# Orbit Integration (matching PME's OrbitCalculator.jl)
# ============================================================================

"""
    compute_orbits(config, model, grids; verbose=false) -> OrbitData

Compute orbital frequencies and trajectories using PME's robust approach.
"""
function compute_orbits(config::KalnajsConfig, model, grids::GridData{T};
                        verbose::Bool=false) where T
    NR = config.grid.NR
    Ne = config.grid.Ne
    nw = config.grid.nw
    nwa = config.grid.nwa
    
    # Angular grid
    eps_w = T(1e-8)
    w = T.(range(eps_w, T(π) - eps_w, length=nw))
    dw = w[2] - w[1]
    cosw = cos.(w)
    sinw = sin.(w)
    
    nwai = (nw - 1) ÷ (nwa - 1)
    Ia = 1:nwai:nw
    wa = w[Ia]
    
    # Output arrays
    Omega_1 = zeros(T, NR, Ne)
    Omega_2 = zeros(T, NR, Ne)
    w1 = fill(T(NaN), nwa, NR, Ne)
    ra = fill(T(NaN), nwa, NR, Ne)
    pha = fill(T(NaN), nwa, NR, Ne)
    jacobian = zeros(T, NR, Ne)
    
    # Main orbit integration loop
    for iR in 1:NR
        R_i = grids.Rc[iR, 1]
        kappa_iE = Models.epicyclic_frequency(model, R_i)
        Omega_iE = Models.rotation_frequency(model, R_i)
        
        for ie in 1:Ne
            E_i = grids.E[iR, ie]
            L_j = grids.L_m[iR, ie]
            L2_j = grids.L2_m[iR, ie]
            ecc = grids.e[iR, ie]
            r1 = grids.R1[iR, ie]
            r2 = grids.R2[iR, ie]
            
            # Circular orbit limit
            if abs(ecc) < T(1e-10)
                Omega_1[iR, ie] = kappa_iE
                Omega_2[iR, ie] = Omega_iE * grids.SGNL[iR, ie]
                w1[:, iR, ie] .= wa
                ra[:, iR, ie] .= R_i
                pha[:, iR, ie] .= wa .* Omega_2[iR, ie] / Omega_1[iR, ie]
                continue
            end
            
            # Near-circular epicyclic approximation
            if ecc < T(0.01)
                a_j = (r2 - r1) / 2
                Omega_1[iR, ie] = kappa_iE
                Omega_2[iR, ie] = Omega_iE * grids.SGNL[iR, ie]
                w1[:, iR, ie] .= wa
                ra[:, iR, ie] .= R_i .- a_j .* cosw[Ia]
                pha[:, iR, ie] .= wa .* Omega_2[iR, ie] / Omega_1[iR, ie] .+
                    2 * Omega_iE / kappa_iE * a_j / R_i .* sinw[Ia] .* grids.SGNL[iR, ie]
                continue
            end
            
            # General orbit integration (MATLAB approach - no fallback needed)
            rs = (r1 + r2) / 2
            drs = (r2 - r1) / 2
            xs = rs .- drs .* cosw
            
            # Compute rvr and svr directly like MATLAB
            rvr = zeros(T, nw)
            svr = zeros(T, nw)
            
            if L2_j > T(1e-12)
                # Non-radial orbit
                for i in eachindex(xs)
                    x = xs[i]
                    V_x = Models.potential(model, x)
                    vr_sq = T(2) * (E_i - V_x) * x^2 - L2_j
                    rvr[i] = sqrt(max(T(0), vr_sq))
                end
                svr .= sinw .* xs ./ rvr
                # Endpoint extrapolation (MATLAB style)
                svr[1] = T(2) * svr[2] - svr[3]
                svr[end] = T(2) * svr[end-1] - svr[end-2]
            else
                # Radial orbit case (L=0)
                for i in eachindex(xs)
                    x = xs[i]
                    V_x = Models.potential(model, x)
                    vr_sq = T(2) * (E_i - V_x)
                    rvr[i] = sqrt(max(T(0), vr_sq))
                end
                svr .= sinw ./ rvr
                svr[end] = T(2) * svr[end-1] - svr[end-2]
            end
            
            # Integrate for radial period
            dt1 = drs .* dw .* svr
            dt2 = zeros(T, nw)
            dt2[2:end] .= (dt1[1:end-1] .+ dt1[2:end]) ./ 2
            t = cumsum(dt2)
            Omega_1[iR, ie] = T(π) / t[end]
            
            w1[:, iR, ie] .= t[Ia] .* Omega_1[iR, ie]
            ra[:, iR, ie] .= xs[Ia]
            
            # Azimuthal phase integration
            if abs(T(1) - ecc) > T(1e-10) && L2_j > T(1e-12)
                svr_phi = sinw ./ rvr
                svr_phi[1] = T(2) * svr_phi[2] - svr_phi[3]
                svr_phi[end] = T(2) * svr_phi[end-1] - svr_phi[end-2]
                dt3 = drs .* dw .* svr_phi ./ xs
                dt4 = zeros(T, nw)
                dt4[2:end] .= (dt3[1:end-1] .+ dt3[2:end]) ./ 2
                phi_arr = cumsum(dt4)
                ph = L_j .* phi_arr[Ia]
            else
                ph = T(π/2) .* ones(T, length(Ia)) .* grids.SGNL[iR, ie]
                ph[1] = zero(T)
            end
            
            Omega_2[iR, ie] = Omega_1[iR, ie] * ph[end] / T(π)
            pha[:, iR, ie] .= ph
        end
        
        if verbose && mod(iR, 20) == 0
            print(".")
        end
    end
    
    if verbose
        println(" done.")
    end
    
    # Compute Jacobian
    compute_jacobian!(jacobian, grids, Omega_1, model)
    
    # Compute distribution function values
    F0 = zeros(T, NR, Ne)
    FE = zeros(T, NR, Ne)
    FL = zeros(T, NR, Ne)
    
    for iR in 1:NR
        for ie in 1:Ne
            F0[iR, ie] = Models.distribution_function(model, grids.E[iR, ie], grids.L_m[iR, ie])
            FE[iR, ie] = Models.df_energy_derivative(model, grids.E[iR, ie], grids.L_m[iR, ie])
            FL[iR, ie] = Models.df_angular_derivative(model, grids.E[iR, ie], grids.L_m[iR, ie])
        end
    end
    
    # Handle NaN/Inf
    F0[.!isfinite.(F0)] .= zero(T)
    FE[.!isfinite.(FE)] .= zero(T)
    FL[.!isfinite.(FL)] .= zero(T)
    
    return OrbitData{T}(grids, Omega_1, Omega_2, w1, ra, pha, jacobian, F0, FE, FL)
end

"""
Compute Jacobian for the (Rc, v) → (E, L) transformation.
"""
function compute_jacobian!(jacobian::Matrix{T}, grids::GridData{T}, 
                          Omega_1::Matrix{T}, model) where T
    NR, Ne = size(jacobian)
    
    for iR in 1:NR
        for ie in 1:Ne
            r1 = grids.R1[iR, ie]
            r2 = grids.R2[iR, ie]
            rc = grids.Rc[iR, ie]
            E_val = grids.E[iR, ie]
            L_val = grids.L_m[iR, ie]
            omega1 = Omega_1[iR, ie]
            
            jacobian[iR, ie] = zero(T)
            
            # General orbit (not circular, not radial)
            general_orbit = (abs(r2 - r1) > T(1e-12)) && (abs(L_val) > T(1e-12))
            if general_orbit
                V1 = Models.potential(model, r1)
                V2 = Models.potential(model, r2)
                dV1 = Models.potential_derivative(model, r1)
                dV2 = Models.potential_derivative(model, r2)
                
                t1 = T(2) * (E_val - V1) * r1 - dV1 * r1^2
                t2 = T(2) * (E_val - V2) * r2 - dV2 * r2^2
                denominator = r2^2 - r1^2
                det_jac = abs(t1 * t2 / denominator)
                jacobian[iR, ie] = T(2) * det_jac * rc / omega1 / abs(L_val)
            end
            
            # Radial orbit
            radial_orbit = abs(L_val) <= T(1e-12)
            if radial_orbit
                DelE = T(2) * (E_val - Models.potential(model, r1))
                jacobian[iR, ie] = sqrt(max(DelE, T(0))) * Models.potential_derivative(model, r2) * T(2) * rc / omega1
                if !isfinite(jacobian[iR, ie])
                    jacobian[iR, ie] = zero(T)
                end
            end
        end
    end
end

end # module OrbitIntegration
