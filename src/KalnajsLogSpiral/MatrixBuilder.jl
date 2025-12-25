# src/KalnajsLogSpiral/MatrixBuilder.jl
"""
Matrix assembly and determinant computation for Kalnajs log-spiral eigenvalue problem.

Builds M(β,α;ω) and computes det(I - M·dα) with GPU acceleration.
"""
module MatrixBuilder

using LinearAlgebra
using ..Configuration
using ..GPUBackend
using ..BasisFunctions

export compute_determinant, compute_determinant_batched
export build_M_matrix  # For debugging

# ============================================================================
# CPU Implementation
# ============================================================================

"""
    build_M_matrix_cpu(omega::Complex, precomputed, Tcpu::Type) -> Matrix{Complex{Tcpu}}

Build M matrix on CPU in the specified precision.
M[β,α] = G × Σ_l W_l^T(β) × diag(DJ × F0l / (ω - Ω_res)) × W_l^*(α) × N_kernel(α)
"""
function build_M_matrix_cpu(omega::Complex{T}, precomputed::PrecomputedData{T}, 
                            Tcpu::Type) where T
    N_alpha = precomputed.N_alpha
    n_l = precomputed.n_l
    NPh = precomputed.NPh
    G = Tcpu(precomputed.G)
    
    CT = Complex{Tcpu}
    M = zeros(CT, N_alpha, N_alpha)
    omega_tcpu = CT(omega)
    
    # Reshape W_l for easier indexing: W_l_mat is [NPh, N_alpha, n_l]
    for i_l in 1:n_l
        # Extract W_l[:, :, i_l] for this harmonic
        W = CT.(precomputed.W_l_mat[:, :, i_l])  # [NPh, N_alpha] (complex)
        
        # Compute denominator: ω - Ω_res
        denom = omega_tcpu .- CT.(precomputed.Omega_res[:, i_l])
        
        # Avoid near-zero denominators
        small_idx = abs.(denom) .< Tcpu(1e-12)
        denom[small_idx] .= CT(1e-12 + 1e-12im)
        
        # Weight: DJ × F0l / (ω - Ω_res)
        weight = Tcpu.(precomputed.DJ_vec) .* Tcpu.(precomputed.F0l_all[:, i_l]) ./ denom
        
        # Accumulate: M += W^T × diag(weight) × conj(W)
        # This is: M += transpose(W) * (weight .* conj.(W))
        for α in 1:N_alpha
            for β in 1:N_alpha
                M[β, α] += sum(W[:, β] .* weight .* conj.(W[:, α]))
            end
        end
    end
    
    # Apply G and N_kernel scaling
    M .*= G
    for α in 1:N_alpha
        M[:, α] .*= Tcpu(precomputed.N_kernel[α])
    end
    
    return M
end

"""
    compute_determinant_cpu(omega::Complex, precomputed, Tcpu::Type) -> Complex{Tcpu}

Compute det(I - M·dα) on CPU in the specified precision.
"""
function compute_determinant_cpu(omega::Complex{T}, precomputed::PrecomputedData{T},
                                 Tcpu::Type) where T
    M = build_M_matrix_cpu(omega, precomputed, Tcpu)
    
    N_alpha = precomputed.N_alpha
    CT = Complex{Tcpu}
    I_mat = Matrix{CT}(I, N_alpha, N_alpha)
    
    # d_alpha is already included in N_kernel, so just compute det(I - M)
    det_val = det(I_mat - M)
    return det_val
end

# ============================================================================
# GPU Implementation (avoiding scalar indexing)
# ============================================================================

"""
    build_M_matrix_gpu(omega::Complex{Tgpu}, devprecomp::DevicePrecomputed{Tgpu}, 
                       gpu_type) -> Matrix{Complex{Tgpu}} (on device)

Build M matrix on GPU. Returns device array.
All operations avoid scalar indexing by using broadcasting and matrix ops.
"""
function build_M_matrix_gpu(omega::Complex{Tgpu}, devprecomp::DevicePrecomputed{Tgpu,AT},
                           gpu_type) where {Tgpu,AT}
    
    DevArray = GPUBackend.gpu_array_type()
    
    N_alpha = devprecomp.N_alpha
    n_l = devprecomp.n_l
    NPh = devprecomp.NPh
    G = devprecomp.G
    
    CT = Complex{Tgpu}
    M = DevArray(zeros(CT, N_alpha, N_alpha))
    
    # Pre-transfer small arrays to CPU for manipulation, build on CPU, then do GPU math
    DJ_vec = devprecomp.DJ_vec  # CPU array
    F0l_all = devprecomp.F0l_all  # CPU array
    Omega_res = devprecomp.Omega_res  # CPU array
    N_kernel = devprecomp.N_kernel  # CPU array
    
    for i_l in 1:n_l
        # W_l is on device: devprecomp.W_l_mat[:, :, i_l]
        W = devprecomp.W_l_mat[:, :, i_l]  # [NPh, N_alpha] on device
        
        # Compute weight on CPU, then transfer
        denom_cpu = omega .- CT.(Omega_res[:, i_l])
        small_idx = abs.(denom_cpu) .< Tgpu(1e-12)
        denom_cpu[small_idx] .= CT(1e-12 + 1e-12im)
        
        weight_cpu = CT.(DJ_vec) .* CT.(F0l_all[:, i_l]) ./ denom_cpu
        weight_dev = DevArray(weight_cpu)  # [NPh] on device
        
        # GPU computation: W^T × diag(weight) × conj(W)
        # Reshape weight for broadcasting: [NPh, 1]
        weight_col = reshape(weight_dev, :, 1)
        
        # W_weighted = W .* weight (broadcast weight across columns)
        W_weighted = conj.(W) .* weight_col
        W_conj = conj.(W)
        
        # M += W_conj^T * W_weighted
        M .+= transpose(W) * W_weighted
    end
    
    # Apply G scaling
    M .*= G
    
    # Apply N_kernel scaling: M[:, α] *= N_kernel[α]
    # Use row-wise broadcast: M .* N_kernel' (N_kernel as row vector)
    N_kernel_row = DevArray(reshape(CT.(N_kernel), 1, :))  # [1, N_alpha] on device
    M .*= N_kernel_row
    
    return M
end

"""
    compute_determinant_gpu(omega::Complex, precomputed::PrecomputedData,
                           devprecomp::DevicePrecomputed{Tgpu}, 
                           gpu_type, Tcpu::Type) -> Complex{Tcpu}

Compute det(I - M·dα) using GPU for M assembly, then transfer to CPU for determinant.
Returns result in Tcpu precision.
"""
function compute_determinant_gpu(omega::Complex{T}, precomputed::PrecomputedData{T},
                                devprecomp::DevicePrecomputed{Tgpu,AT},
                                gpu_type, Tcpu::Type) where {T,Tgpu,AT}
    
    # Convert omega to GPU precision
    omega_gpu = Complex{Tgpu}(omega)
    
    # Build M on GPU
    M_gpu = build_M_matrix_gpu(omega_gpu, devprecomp, gpu_type)
    
    # Transfer M to host
    M_host = Array(M_gpu)
    
    # Convert to CPU precision
    M_cpu = Complex{Tcpu}.(M_host)
    
    # d_alpha is already included in N_kernel, so just compute det(I - M)
    N_alpha = precomputed.N_alpha
    CT = Complex{Tcpu}
    I_mat = Matrix{CT}(I, N_alpha, N_alpha)
    det_val = det(I_mat - M_cpu)
    
    return det_val
end

# ============================================================================
# Unified Interface
# ============================================================================

"""
    compute_determinant(omega, precomputed, devprecomp_or_nothing, 
                       gpu_type, Tcpu) -> Complex{Tcpu}

Unified determinant computation. Routes to GPU or CPU based on devprecomp.
"""
function compute_determinant(omega::Complex{T}, precomputed::PrecomputedData{T},
                            devprecomp::Union{Nothing,DevicePrecomputed},
                            gpu_type, Tcpu::Type) where T
    if devprecomp === nothing || gpu_type == GPUBackend.NONE
        # CPU path
        return compute_determinant_cpu(omega, precomputed, Tcpu)
    else
        # GPU path
        return compute_determinant_gpu(omega, precomputed, devprecomp, gpu_type, Tcpu)
    end
end

"""
    compute_determinant_batched(omegas, precomputed, devprecomp_or_nothing,
                               gpu_type, Tcpu; batch_size=100) -> Vector{Complex{Tcpu}}

Batched determinant computation for multiple omegas.
"""
function compute_determinant_batched(omegas::Vector{Complex{T}}, 
                                    precomputed::PrecomputedData{T},
                                    devprecomp::Union{Nothing,DevicePrecomputed},
                                    gpu_type, Tcpu::Type;
                                    batch_size::Int=100) where T
    n_omega = length(omegas)
    dets = Vector{Complex{Tcpu}}(undef, n_omega)
    
    # Process in batches to avoid memory issues
    for i_start in 1:batch_size:n_omega
        i_end = min(i_start + batch_size - 1, n_omega)
        
        for i in i_start:i_end
            dets[i] = compute_determinant(omegas[i], precomputed, devprecomp, 
                                         gpu_type, Tcpu)
        end
        
        # Synchronize after each batch if on GPU
        if gpu_type != GPUBackend.NONE
            GPUBackend.gpu_synchronize()
        end
    end
    
    return dets
end

# For backward compatibility and debugging
function build_M_matrix(omega::Complex{T}, precomputed::PrecomputedData{T}) where T
    return build_M_matrix_cpu(omega, precomputed, T)
end

end # module MatrixBuilder
