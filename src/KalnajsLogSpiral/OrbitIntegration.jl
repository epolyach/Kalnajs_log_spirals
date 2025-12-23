# src/KalnajsLogSpiral/OrbitIntegration.jl
"""
Orbit integration for log-spiral eigenvalue calculation.

Computes orbital quantities on (Rc, v) phase space grid:
- Frequencies Ω₁ (radial), Ω₂ (azimuthal)
- Orbit trajectories: r(w₁), θ(w₁)
- Jacobian for integration measure

Based on NL_precompute.m orbit integration section.
"""
module OrbitIntegration

using ..Configuration
using ..Models

export OrbitData, compute_orbits, compute_grids

# ============================================================================
# Data Structures
# ============================================================================

"""
Phase space grid arrays
"""
struct PhaseSpaceGrids{T<:AbstractFloat}
    # Grid coordinates
    Rc::Matrix{T}      # Guiding center radius [NR, Ne]
    v::Matrix{T}       # Circulation parameter [NR, Ne]
    
    # Derived quantities
    e::Matrix{T}       # Eccentricity = 1 - |v|
    R1::Matrix{T}      # Pericenter radius
    R2::Matrix{T}      # Apocenter radius
    E::Matrix{T}       # Energy
    L_m::Matrix{T}     # Angular momentum (signed)
    L2_m::Matrix{T}    # Angular momentum squared
    SGNL::Matrix{T}    # Sign of angular momentum
    
    # Integration weights
    S_RC::Vector{T}    # Radial integration weights (logarithmic)
    S_e::Matrix{T}     # Eccentricity integration weights (Simpson)
end

"""
Orbit data arrays
"""
struct OrbitData{T<:AbstractFloat}
    # Grids
    grids::PhaseSpaceGrids{T}
    
    # Frequencies [NR, Ne]
    Omega_1::Matrix{T}  # Radial frequency
    Omega_2::Matrix{T}  # Azimuthal frequency
    
    # Orbit trajectories [nwa, NR, Ne]
    w1::Array{T,3}      # Angle variable (radial phase)
    ra::Array{T,3}      # Radius along orbit
    pha::Array{T,3}     # Azimuthal phase
    
    # Jacobian [NR, Ne]
    jacobian::Matrix{T}
    
    # Distribution function values [NR, Ne]
    F0::Matrix{T}       # DF(E, L)
    FE::Matrix{T}       # ∂DF/∂E
    FL::Matrix{T}       # ∂DF/∂L
end

# ============================================================================
# Grid Construction
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
Compute Simpson integration coefficients
"""
function simpson_coef(x::AbstractVector{T}) where T
    n = length(x)
    if n < 3
        return trapezoid_coef(x)
    end
    
    dx = x[2] - x[1]
    S = zeros(T, n)
    
    if mod(n, 2) == 1  # Odd number of points
        S[1] = dx / 3
        S[end] = dx / 3
        for i in 2:2:n-1
            S[i] = 4 * dx / 3
        end
        for i in 3:2:n-2
            S[i] = 2 * dx / 3
        end
    else
        # Even: fallback to trapezoidal
        S = trapezoid_coef(x)
    end
    
    return S
end

"""
    compute_grids(config::KalnajsConfig, model::ToomreZangModel) -> PhaseSpaceGrids

Construct phase space grids in (Rc, v) coordinates.
"""
function compute_grids(config::KalnajsConfig, model::ToomreZangModel{T}) where T
    NR = config.grid.NR
    Ne = config.grid.Ne
    Rc_min = T(config.grid.Rc_min)
    Rc_max = T(config.grid.Rc_max)
    v_min = T(config.grid.v_min)
    
    # Logarithmic radial grid
    Rc_vec = T.(exp.(range(log(Rc_min), log(Rc_max), length=NR)))
    uRc = log.(Rc_vec)
    S_RC = Rc_vec .* trapezoid_coef(uRc)
    
    # Circulation grid with Simpson weights
    S_e = zeros(T, NR, Ne)
    v_grid = zeros(T, NR, Ne)
    
    for iR in 1:NR
        v_vec = T.(range(v_min, 1, length=Ne))
        S_e[iR, :] = simpson_coef(v_vec)
        v_grid[iR, :] = v_vec
    end
    
    # Avoid exactly zero v
    v_grid[abs.(v_grid) .< T(1e-14)] .= T(1e-12)
    SGNL = sign.(v_grid)
    
    # Eccentricity and orbital radii
    e = one(T) .- abs.(v_grid)
    Rc = repeat(Rc_vec, 1, Ne)
    R1 = Rc .* (one(T) .- e)
    R2 = Rc .* (one(T) .+ e)
    
    # Energy from pericenter/apocenter
    E = similar(Rc)
    for iR in 1:NR
        for ie in 1:Ne
            E[iR, ie] = Models.compute_energy(model, R1[iR, ie], R2[iR, ie])
        end
    end
    
    # Handle singular cases
    bad_E = .!isfinite.(E) .& (e .< one(T))
    E[bad_E] .= Rc[bad_E] .* Models.potential_derivative.(Ref(model), Rc[bad_E]) ./ 2 .+ 
                Models.potential.(Ref(model), Rc[bad_E])
    
    bad_E2 = .!isfinite.(E)
    E[bad_E2] .= Models.potential.(Ref(model), R2[bad_E2])
    
    # Angular momentum
    L2_m = T(2) .* (E .- Models.potential.(Ref(model), R2)) .* R2.^2
    L2_m[L2_m .< T(1e-14)] .= zero(T)
    L_m = sqrt.(L2_m) .* SGNL
    
    return PhaseSpaceGrids{T}(Rc, v_grid, e, R1, R2, E, L_m, L2_m, SGNL, S_RC, S_e)
end

# ============================================================================
# Orbit Integration
# ============================================================================

"""
    compute_orbits(config::KalnajsConfig, model::ToomreZangModel, grids::PhaseSpaceGrids) -> OrbitData

Compute orbital trajectories and frequencies.

This implements the orbit integration from NL_precompute.m lines 106-200.
"""
function compute_orbits(config::KalnajsConfig, model::ToomreZangModel{T}, 
                        grids::PhaseSpaceGrids{T}; verbose::Bool=false) where T
    
    NR = config.grid.NR
    Ne = config.grid.Ne
    nw = config.grid.nw
    nwa = config.grid.nwa
    
    # Orbit integration angle
    eps_w = T(1e-8)
    w_full = T.(range(eps_w, T(π) - eps_w, length=nw))
    dw = w_full[2] - w_full[1]
    
    # Reduced sampling indices
    nwai = div(nw - 1, nwa - 1)
    Ia = 1:nwai:nw
    wa = w_full[Ia]
    
    cosw = cos.(w_full)
    sinw = sin.(w_full)
    
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
            
            # General orbit integration
            rs = (r1 + r2) / 2
            drs = (r2 - r1) / 2
            xs = rs .- drs .* cosw
            
            # Radial velocity integrand
            if L2_j > T(1e-12)
                rvr = sqrt.(T(2) .* (E_i .- Models.potential.(Ref(model), xs)) .* xs.^2 .- L2_j)
                svr = sinw .* xs ./ rvr
                svr[1] = 2 * svr[2] - svr[3]
                svr[end] = 2 * svr[end-1] - svr[end-2]
            else
                vr = sqrt.(T(2) .* (E_i .- Models.potential.(Ref(model), xs)))
                svr = sinw ./ vr
                svr[end] = 2 * svr[end-1] - svr[end-2]
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
            if abs(1 - ecc) > T(1e-10) && L2_j > T(1e-12)
                svr_phi = sinw ./ rvr
                svr_phi[1] = 2 * svr_phi[2] - svr_phi[3]
                svr_phi[end] = 2 * svr_phi[end-1] - svr_phi[end-2]
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
    compute_jacobian!(jacobian, grids, model, Omega_1)
    
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

From NL_precompute.m lines 203-243.
"""
function compute_jacobian!(jacobian::Matrix{T}, grids::PhaseSpaceGrids{T}, 
                          model::ToomreZangModel{T}, Omega_1::Matrix{T}) where T
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
            
            # General orbit case
            if abs(r2 - r1) > T(1e-12) && abs(L_val) > T(1e-12)
                V1 = Models.potential(model, r1)
                V2 = Models.potential(model, r2)
                dV1 = Models.potential_derivative(model, r1)
                dV2 = Models.potential_derivative(model, r2)
                
                t1 = 2 * (E_val - V1) * r1 - dV1 * r1^2
                t2 = 2 * (E_val - V2) * r2 - dV2 * r2^2
                denominator = r2^2 - r1^2
                det_jac = abs(t1 * t2 / denominator)
                jacobian[iR, ie] = 2 * det_jac * rc / omega1 / abs(L_val)
            end
            
            # Radial orbit case
            if abs(L_val) <= T(1e-12)
                DelE = T(2) * (E_val - Models.potential(model, r1))
                jacobian[iR, ie] = sqrt(max(DelE, zero(T))) * Models.potential_derivative(model, r2) * 
                                   2 * rc / omega1
                if !isfinite(jacobian[iR, ie])
                    jacobian[iR, ie] = zero(T)
                end
            end
        end
    end
end

end # module OrbitIntegration
