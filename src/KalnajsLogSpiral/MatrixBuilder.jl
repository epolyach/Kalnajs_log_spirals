# src/KalnajsLogSpiral/MatrixBuilder.jl
"""
M(ω) matrix construction for log-spiral eigenvalue problem.

Builds the matrix M(β,α;ω) from Eq. 45:
M = G × Σ_l W_l^T × diag(DJ × F0l / (ω - Ω_res)) × W_l^* × N_kernel

Then computes det(I - M·dα) for eigenvalue search.
"""
module MatrixBuilder

using LinearAlgebra
using ..Configuration
using ..GPUBackend
using ..BasisFunctions

export build_M_matrix, compute_determinant

# ============================================================================
# M Matrix Construction
# ============================================================================

"""
    build_M_matrix(omega::Complex, precomputed::PrecomputedData, backend) -> Matrix{Complex}

Build the M(β,α;ω) matrix from Eq. 45.

M[β,α] = G × Σ_l W_l[β]^T × diag(DJ × F0l / (ω - Ω_res)) × W_l[α]^*

where the sum is over radial harmonics l.

# Arguments
- `omega`: Complex frequency ω = m×Ω_p + i×γ
- `precomputed`: Precomputed ω-independent data
- `backend`: GPU backend type

# Returns
- M matrix of size [N_alpha, N_alpha]
"""
function build_M_matrix(omega::Complex{T}, precomputed::PrecomputedData{T},
                        backend::GPUBackend.GPUBackendType) where T
    
    N_alpha = precomputed.N_alpha
    n_l = precomputed.n_l
    NPh = precomputed.NPh
    G = precomputed.G
    
    # Initialize M matrix
    CT = Complex{T}
    M = zeros(CT, N_alpha, N_alpha)
    
    # Loop over radial harmonics
    for i_l in 1:n_l
        # Resonance denominator: ω - (l×Ω₁ + m×Ω₂)
        denom = omega .- precomputed.Omega_res[:, i_l]
        
        # Avoid division by zero near resonances
        small_idx = abs.(denom) .< T(1e-12)
        denom[small_idx] .= CT(1e-12 + 1e-12im)
        
        # Weight vector: DJ × F0l / denom
        weight = precomputed.DJ_vec .* precomputed.F0l_all[:, i_l] ./ denom
        
        # W_l matrix for this harmonic [NPh, N_alpha]
        W = precomputed.W_l_mat[:, :, i_l]
        
        # M += W^T × diag(weight) × conj(W)
        # This is: M[β,α] = Σ_J weight[J] × W[J,β] × W[J,α]*
        M .+= transpose(W) * (weight .* conj.(W))
    end
    
    # Apply gravitational constant and N_kernel
    M .*= G
    
    # Apply N_kernel as diagonal scaling: M[β,α] *= N_kernel[α]
    for α in 1:N_alpha
        M[:, α] .*= precomputed.N_kernel[α]
    end
    
    return M
end

"""
    build_M_matrix_batched(omegas::Vector{Complex}, precomputed, backend) -> Vector{Matrix}

Build M matrices for multiple ω values (for grid scan).

This can be parallelized across GPUs.
"""
function build_M_matrix_batched(omegas::Vector{Complex{T}}, precomputed::PrecomputedData{T},
                                backend::GPUBackend.GPUBackendType) where T
    
    n_omega = length(omegas)
    M_matrices = Vector{Matrix{Complex{T}}}(undef, n_omega)
    
    # Simple loop - can be parallelized with @threads or distributed
    Threads.@threads for i in 1:n_omega
        M_matrices[i] = build_M_matrix(omegas[i], precomputed, backend)
    end
    
    return M_matrices
end

# ============================================================================
# Determinant Computation
# ============================================================================

"""
    compute_determinant(omega::Complex, precomputed, backend) -> Complex

Compute det(I - M·dα) for the given ω.

The eigenvalue equation is det(I - M) = 0 where M already includes dα integration weights.
"""
function compute_determinant(omega::Complex{T}, precomputed::PrecomputedData{T},
                             backend::GPUBackend.GPUBackendType) where T
    
    M = build_M_matrix(omega, precomputed, backend)
    N_alpha = precomputed.N_alpha
    
    # Compute det(I - M)
    I_mat = Matrix{Complex{T}}(I, N_alpha, N_alpha)
    det_val = det(I_mat - M)
    
    return det_val
end

"""
    compute_determinant_batched(omegas, precomputed, backend) -> Vector{Complex}

Compute determinants for multiple ω values.
"""
function compute_determinant_batched(omegas::Vector{Complex{T}}, precomputed::PrecomputedData{T},
                                     backend::GPUBackend.GPUBackendType) where T
    
    n_omega = length(omegas)
    det_vals = Vector{Complex{T}}(undef, n_omega)
    
    Threads.@threads for i in 1:n_omega
        det_vals[i] = compute_determinant(omegas[i], precomputed, backend)
    end
    
    return det_vals
end

# ============================================================================
# Condition Number Monitoring
# ============================================================================

"""
    compute_determinant_with_condition(omega, precomputed, backend) -> (det, cond_num)

Compute determinant and condition number for numerical stability monitoring.
"""
function compute_determinant_with_condition(omega::Complex{T}, precomputed::PrecomputedData{T},
                                            backend::GPUBackend.GPUBackendType) where T
    
    M = build_M_matrix(omega, precomputed, backend)
    N_alpha = precomputed.N_alpha
    
    I_mat = Matrix{Complex{T}}(I, N_alpha, N_alpha)
    A = I_mat - M
    
    det_val = det(A)
    cond_num = cond(A)
    
    return (det_val, cond_num)
end

end # module MatrixBuilder
