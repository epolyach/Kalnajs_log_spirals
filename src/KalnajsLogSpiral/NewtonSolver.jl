# src/KalnajsLogSpiral/NewtonSolver.jl
"""
Newton-Raphson solver for eigenvalue refinement.
"""
module NewtonSolver

using Printf
using LinearAlgebra
using ..Configuration
using ..BasisFunctions
using ..MatrixBuilder
using ..GPUBackend

export newton_search, NewtonResult

"""
Result structure for Newton solver
"""
struct NewtonResult{T<:AbstractFloat}
    Omega_p::T
    gamma::T
    omega::Complex{T}
    det_val::Complex{T}
    iterations::Int
    converged::Bool
    history::Matrix{T}  # [iteration, (Omega_p, gamma, |det|)]
end

"""
    newton_search(config, precomputed, devprecomp, gpu_type, Tcpu,
                 Omega_p_init, gamma_init; verbose=false) -> NewtonResult

Newton-Raphson search for eigenvalue starting from initial guess.
Uses finite difference Jacobian.
"""
function newton_search(config::KalnajsConfig,
                      precomputed::PrecomputedData{T},
                      devprecomp::Union{Nothing,DevicePrecomputed},
                      gpu_type,
                      Tcpu::Type,
                      Omega_p_init::Real,
                      gamma_init::Real;
                      verbose::Bool=false) where T
    
    # Newton parameters
    max_iter = config.newton.max_iter
    tol = Tcpu(config.newton.tol)
    delta = Tcpu(config.newton.delta)
    use_line_search = config.newton.use_line_search
    max_line_search = config.newton.max_line_search
    m = config.physics.m
    
    # Initialize
    Omega_p = Tcpu(Omega_p_init)
    gamma_val = Tcpu(gamma_init)
    history = zeros(Tcpu, max_iter + 1, 3)
    
    converged = false
    iteration = 0
    
    if verbose
        println("Newton-Raphson refinement:")
        println("  max_iter = $max_iter, tol = $tol, delta = $delta")
        println()
    end
    
    for iter in 1:max_iter
        iteration = iter
        
        # Current omega
        omega = Complex{T}(m * Omega_p, gamma_val)
        
        # Evaluate determinant
        det_val = MatrixBuilder.compute_determinant(omega, precomputed, devprecomp, gpu_type, Tcpu)
        
        # Residual: F = [Re(det), Im(det)]
        F = Tcpu[real(det_val), imag(det_val)]
        res_norm = norm(F)
        
        # Store history
        history[iter, :] = [Omega_p, gamma_val, abs(det_val)]
        
        if verbose
            @printf("Iter %2d: Ω_p=%.10f  γ=%.10f  det=%.2e%+.2ei  |F|=%.2e\n",
                   iter, Omega_p, gamma_val, real(det_val), imag(det_val), res_norm)
        end
        
        # Check convergence
        if res_norm < tol
            converged = true
            if verbose
                println("✓ Converged!")
            end
            break
        end
        
        # Compute Jacobian via finite differences
        omega_Op = Complex{T}(m * (Omega_p + delta), gamma_val)
        omega_g = Complex{T}(m * Omega_p, gamma_val + delta)
        
        det_Op = MatrixBuilder.compute_determinant(omega_Op, precomputed, devprecomp, gpu_type, Tcpu)
        det_g = MatrixBuilder.compute_determinant(omega_g, precomputed, devprecomp, gpu_type, Tcpu)
        
        # Jacobian: J[i,j] = ∂F_i/∂x_j where x = [Omega_p, gamma]
        J = zeros(Tcpu, 2, 2)
        J[1, 1] = (real(det_Op) - real(det_val)) / delta  # ∂Re(det)/∂Ω_p
        J[1, 2] = (real(det_g) - real(det_val)) / delta   # ∂Re(det)/∂γ
        J[2, 1] = (imag(det_Op) - imag(det_val)) / delta  # ∂Im(det)/∂Ω_p
        J[2, 2] = (imag(det_g) - imag(det_val)) / delta   # ∂Im(det)/∂γ
        
        # Newton step: Δx = -J^{-1} F
        delta_x = -J \ F
        
        # Line search if enabled
        if use_line_search
            alpha = Tcpu(1.0)
            for ls_iter in 1:max_line_search
                Op_new = Omega_p + alpha * delta_x[1]
                gamma_new = gamma_val + alpha * delta_x[2]
                
                # Check bounds
                if gamma_new < Tcpu(1e-4) || Op_new < Tcpu(0) || Op_new > Tcpu(1)
                    alpha *= Tcpu(0.5)
                    continue
                end
                
                # Evaluate at new point
                omega_new = Complex{T}(m * Op_new, gamma_new)
                det_new = MatrixBuilder.compute_determinant(omega_new, precomputed, devprecomp, gpu_type, Tcpu)
                F_new = Tcpu[real(det_new), imag(det_new)]
                res_new = norm(F_new)
                
                # Accept if residual decreases
                if res_new < res_norm
                    Omega_p = Op_new
                    gamma_val = gamma_new
                    break
                end
                
                alpha *= Tcpu(0.5)
                
                if ls_iter == max_line_search
                    # Line search failed, take full step anyway
                    Omega_p += delta_x[1]
                    gamma_val += delta_x[2]
                end
            end
        else
            # No line search, take full Newton step
            Omega_p += delta_x[1]
            gamma_val += delta_x[2]
        end
        
        # Enforce bounds
        gamma_val = max(gamma_val, Tcpu(1e-4))
        Omega_p = clamp(Omega_p, Tcpu(0), Tcpu(1))
    end
    
    # Final evaluation
    omega_final = Complex{T}(m * Omega_p, gamma_val)
    det_final = MatrixBuilder.compute_determinant(omega_final, precomputed, devprecomp, gpu_type, Tcpu)
    
    return NewtonResult{Tcpu}(
        Omega_p,
        gamma_val,
        Complex{Tcpu}(omega_final),
        det_final,
        iteration,
        converged,
        history[1:iteration, :]
    )
end

end # module NewtonSolver
