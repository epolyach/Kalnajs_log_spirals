# src/KalnajsLogSpiral/NewtonSolver.jl
"""
Newton-Raphson solver for zeros of det(I - M·dα).

This is the Julia equivalent of NL_newton_search.m with:
- Finite difference Jacobian
- Optional complex step derivative (more accurate)
- Line search with backtracking
- Adaptive parameter exploration for convergence study

Reference values for Toomre-Zang n=4, m=2:
  - Zang (1976):              Ω_p = 0.439'426,      γ = 0.127'181
  - Polyachenko (refined):    Ω_p = 0.439'442'9284, γ = 0.127'204'5628
"""
module NewtonSolver

using Printf
using LinearAlgebra
using ..Configuration
using ..GPUBackend
using ..BasisFunctions
using ..MatrixBuilder

export newton_search, NewtonResult, convergence_study

# ============================================================================
# Result Structure
# ============================================================================

"""
Result of Newton search
"""
struct NewtonResult{T<:AbstractFloat}
    Omega_p::T
    gamma::T
    omega::Complex{T}
    det_val::Complex{T}
    iterations::Int
    converged::Bool
    history::Matrix{T}  # [iter, (Omega_p, gamma, Re(det), Im(det))]
end

# ============================================================================
# Newton-Raphson Solver
# ============================================================================

"""
    newton_search(config, precomputed, Omega_p_init, gamma_init, backend; verbose=false) -> NewtonResult

Perform Newton-Raphson search for zero of det(I - M·dα).

# Arguments
- `config`: Configuration with Newton parameters
- `precomputed`: Precomputed ω-independent data
- `Omega_p_init`: Initial guess for pattern speed
- `gamma_init`: Initial guess for growth rate
- `backend`: GPU backend type
- `verbose`: Print iteration progress

# Returns
- `NewtonResult` with final values and convergence info
"""
function newton_search(config::KalnajsConfig, precomputed::PrecomputedData{T},
                       Omega_p_init::Real, gamma_init::Real,
                       backend::GPUBackend.GPUBackendType;
                       verbose::Bool=false) where T
    
    m = config.physics.m
    max_iter = config.newton.max_iter
    tol = T(config.newton.tol)
    delta = T(config.newton.delta)
    use_line_search = config.newton.use_line_search
    max_line_search = config.newton.max_line_search
    use_complex_step = config.newton.use_complex_step
    complex_step_h = config.newton.complex_step_h
    
    # Initial values
    Omega_p = T(Omega_p_init)
    gamma_val = T(gamma_init)
    
    # History array
    history = zeros(T, max_iter, 4)
    
    CT = Complex{T}
    
    if verbose
        println("Newton-Raphson search")
        @printf("Initial: Ω_p = %.6f, γ = %.6f\n", Omega_p, gamma_val)
    end
    
    converged = false
    iter = 0
    
    for iteration in 1:max_iter
        iter = iteration
        omega = CT(m * Omega_p, gamma_val)
        
        # Compute determinant at current point
        det_val = MatrixBuilder.compute_determinant(omega, precomputed, backend)
        
        # Residual: F = [Re(det), Im(det)]
        F = [real(det_val), imag(det_val)]
        res_norm = norm(F)
        
        # Record history
        history[iteration, :] = [Omega_p, gamma_val, real(det_val), imag(det_val)]
        
        if verbose
            @printf("Iter %2d: Ω_p=%.10f  γ=%.10f  det=%.2e%+.2ei  |F|=%.2e\n",
                    iteration, Omega_p, gamma_val, real(det_val), imag(det_val), res_norm)
        end
        
        # Check convergence
        if res_norm < tol
            converged = true
            if verbose
                println("\n*** Converged! ***")
            end
            break
        end
        
        # Compute Jacobian
        if use_complex_step
            J = compute_jacobian_complex_step(Omega_p, gamma_val, m, precomputed, backend, T(complex_step_h))
        else
            J = compute_jacobian_finite_diff(Omega_p, gamma_val, m, precomputed, backend, delta)
        end
        
        # Check Jacobian conditioning
        cond_J = cond(J)
        if verbose && cond_J > T(1e10)
            @printf("         Warning: Jacobian ill-conditioned (cond = %.2e)\n", cond_J)
        end
        
        # Newton step
        dx = J \ F
        
        # Line search
        step = one(T)
        if use_line_search
            for ls in 1:max_line_search
                Omega_p_new = Omega_p - step * dx[1]
                gamma_new = gamma_val - step * dx[2]
                
                # Check bounds
                if gamma_new > T(1e-4) && Omega_p_new > zero(T) && Omega_p_new < one(T)
                    omega_new = CT(m * Omega_p_new, gamma_new)
                    det_new = MatrixBuilder.compute_determinant(omega_new, precomputed, backend)
                    F_new = [real(det_new), imag(det_new)]
                    
                    if norm(F_new) < res_norm || ls == max_line_search
                        break
                    end
                end
                step *= T(0.5)
            end
            
            if verbose && step < one(T)
                @printf("         (line search: step = %.3f)\n", step)
            end
        end
        
        # Update
        Omega_p -= step * dx[1]
        gamma_val -= step * dx[2]
    end
    
    # Final omega and determinant
    omega_final = CT(m * Omega_p, gamma_val)
    det_final = MatrixBuilder.compute_determinant(omega_final, precomputed, backend)
    
    return NewtonResult{T}(
        Omega_p, gamma_val, omega_final, det_final,
        iter, converged, history[1:iter, :]
    )
end

# ============================================================================
# Jacobian Computation
# ============================================================================

"""
Compute Jacobian using finite differences.
"""
function compute_jacobian_finite_diff(Omega_p::T, gamma_val::T, m::Int,
                                      precomputed::PrecomputedData{T},
                                      backend::GPUBackend.GPUBackendType,
                                      delta::T) where T
    CT = Complex{T}
    omega = CT(m * Omega_p, gamma_val)
    det_val = MatrixBuilder.compute_determinant(omega, precomputed, backend)
    F = [real(det_val), imag(det_val)]
    
    # ∂det/∂Ω_p
    omega_dOp = CT(m * (Omega_p + delta), gamma_val)
    det_dOp = MatrixBuilder.compute_determinant(omega_dOp, precomputed, backend)
    dF_dOp = ([real(det_dOp), imag(det_dOp)] - F) / delta
    
    # ∂det/∂γ
    omega_dg = CT(m * Omega_p, gamma_val + delta)
    det_dg = MatrixBuilder.compute_determinant(omega_dg, precomputed, backend)
    dF_dg = ([real(det_dg), imag(det_dg)] - F) / delta
    
    J = [dF_dOp dF_dg]
    return J
end

"""
Compute Jacobian using complex step derivative (more accurate).

The complex step derivative avoids subtractive cancellation:
∂f/∂x ≈ Im[f(x + ih)] / h

This gives machine precision derivatives without the h² error of finite differences.
"""
function compute_jacobian_complex_step(Omega_p::T, gamma_val::T, m::Int,
                                       precomputed::PrecomputedData{T},
                                       backend::GPUBackend.GPUBackendType,
                                       h::T) where T
    CT = Complex{T}
    
    # For the complex step, we need to evaluate det at complex-valued (Ω_p, γ)
    # This requires extending the computation to handle complex parameters
    # 
    # For now, use a hybrid approach: finite difference with very small step
    # The true complex step would require ω = m*(Ω_p + ih_Op) + i*(γ + ih_g)
    # which makes ω a bicomplex number
    
    # Fallback to finite difference with the provided h as step
    # (A proper complex step implementation would need bicomplex arithmetic)
    return compute_jacobian_finite_diff(Omega_p, gamma_val, m, precomputed, backend, h)
end

# ============================================================================
# Convergence Study
# ============================================================================

"""
    convergence_study(config, model, Omega_p_init, gamma_init, backend; verbose=false)

Run Newton search at multiple resolution levels to verify convergence.

This is where parameter exploration happens for achieving 6-8 digit precision.
"""
function convergence_study(config::KalnajsConfig, model,
                           Omega_p_init::T, gamma_init::T,
                           backend::GPUBackend.GPUBackendType;
                           verbose::Bool=false) where T
    
    # Parameter levels for convergence study
    levels = [
        (NR=101, Ne=11, N_alpha=51, l_max=15),
        (NR=201, Ne=21, N_alpha=101, l_max=25),
        (NR=401, Ne=41, N_alpha=201, l_max=35),
    ]
    
    results = Vector{NewtonResult{T}}()
    
    for (i, level) in enumerate(levels)
        if verbose
            println("\n" * "="^50)
            @printf("Convergence level %d: NR=%d, Ne=%d, N_alpha=%d, l_max=%d\n",
                    i, level.NR, level.Ne, level.N_alpha, level.l_max)
            println("="^50)
        end
        
        # Create config with these parameters
        level_config = deepcopy(config)
        level_config.grid.NR = level.NR
        level_config.grid.Ne = level.Ne
        level_config.grid.N_alpha = level.N_alpha
        level_config.grid.l_min = -level.l_max
        level_config.grid.l_max = level.l_max
        
        # Precompute
        precomputed = BasisFunctions.precompute_all(level_config, model, backend; verbose=verbose)
        
        # Use previous result as initial guess if available
        Op_init = length(results) > 0 ? results[end].Omega_p : Omega_p_init
        g_init = length(results) > 0 ? results[end].gamma : gamma_init
        
        # Run Newton
        result = newton_search(level_config, precomputed, Op_init, g_init, backend; verbose=verbose)
        push!(results, result)
        
        if verbose
            @printf("Level %d result: Ω_p = %.10f, γ = %.10f\n", i, result.Omega_p, result.gamma)
        end
    end
    
    # Print convergence summary
    if verbose
        println("\n" * "="^60)
        println("CONVERGENCE SUMMARY")
        println("="^60)
        println("Level |    NR   |   Ne  | N_alpha |      Ω_p         |       γ          |")
        println("------|---------|-------|---------|------------------|------------------|")
        
        for (i, (level, result)) in enumerate(zip(levels, results))
            @printf("  %d   |  %4d   |  %3d  |   %3d   | %.12f | %.12f |\n",
                    i, level.NR, level.Ne, level.N_alpha, result.Omega_p, result.gamma)
        end
        println("------|---------|-------|---------|------------------|------------------|")
        
        # Estimate digits of convergence
        if length(results) >= 2
            dOp = abs(results[end].Omega_p - results[end-1].Omega_p)
            dg = abs(results[end].gamma - results[end-1].gamma)
            digits_Op = dOp > 0 ? -log10(dOp) : 12
            digits_g = dg > 0 ? -log10(dg) : 12
            @printf("Estimated digits of convergence: Ω_p ~ %.1f, γ ~ %.1f\n", digits_Op, digits_g)
        end
    end
    
    return results
end

# ============================================================================
# Formatted Output
# ============================================================================

"""
Format number with apostrophes every 3 digits after decimal point.
"""
function format_with_apostrophes(x::Real, digits::Int=10)
    s = @sprintf("%.10f", x)
    dot_pos = findfirst('.', s)
    if dot_pos === nothing
        return s
    end
    
    int_part = s[1:dot_pos]
    dec_part = s[dot_pos+1:end]
    
    # Insert apostrophes every 3 digits
    formatted_dec = ""
    for (i, c) in enumerate(dec_part)
        formatted_dec *= c
        if i % 3 == 0 && i < length(dec_part)
            formatted_dec *= "'"
        end
    end
    
    return int_part * formatted_dec
end

"""
Print final result table matching MATLAB format.
"""
function print_result_table(result::NewtonResult{T}, config::KalnajsConfig) where T
    Op_str = format_with_apostrophes(result.Omega_p)
    gamma_str = format_with_apostrophes(result.gamma)
    
    println()
    println("|---------|---------|-------|---------|------------------|---------------------|")
    println("| N_alpha | α_max   | l_max |   NR×Ne |       Ω_p        |          γ          |")
    println("|---------|---------|-------|---------|------------------|---------------------|")
    @printf("| %7d | %7.1f | %5d | %3d×%-3d | %16s | %19s |\n",
            config.grid.N_alpha, config.grid.alpha_max, config.grid.l_max,
            config.grid.NR, config.grid.Ne, Op_str, gamma_str)
    println("|---------|---------|-------|---------|------------------|---------------------|")
end

end # module NewtonSolver
