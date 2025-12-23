# src/KalnajsLogSpiral/GridScan.jl
"""
Grid scan over complex ω-plane to find approximate zeros of det(I - M·dα).

This is the Julia equivalent of NL_grid_scan.m
"""
module GridScan

using Printf
using ..Configuration
using ..GPUBackend
using ..BasisFunctions
using ..MatrixBuilder

export grid_scan, find_minimum

# ============================================================================
# Grid Scan
# ============================================================================

"""
    grid_scan(config::KalnajsConfig, precomputed::PrecomputedData, backend; verbose=false)

Perform grid scan over (Ω_p, γ) plane to find approximate zero location.

# Arguments
- `config`: Configuration with scan parameters
- `precomputed`: Precomputed ω-independent data
- `backend`: GPU backend type
- `verbose`: Print progress

# Returns
- `det_grid`: Matrix of determinant values [n_gamma, n_Op]
- `Omega_p_init`: Initial guess for Ω_p
- `gamma_init`: Initial guess for γ
"""
function grid_scan(config::KalnajsConfig, precomputed::PrecomputedData{T},
                   backend::GPUBackend.GPUBackendType;
                   verbose::Bool=false) where T
    
    m = config.physics.m
    n_Op = config.scan.n_Op
    n_gamma = config.scan.n_gamma
    Omega_p_min = T(config.scan.Omega_p_min)
    Omega_p_max = T(config.scan.Omega_p_max)
    gamma_min = T(config.scan.gamma_min)
    gamma_max = T(config.scan.gamma_max)
    
    # Create grids
    Omega_p_grid = T.(range(Omega_p_min, Omega_p_max, length=n_Op))
    gamma_grid = T.(range(gamma_min, gamma_max, length=n_gamma))
    
    # Determinant grid
    CT = Complex{T}
    det_grid = zeros(CT, n_gamma, n_Op)
    
    if verbose
        @printf("Scanning %d × %d = %d ω values...\n", n_Op, n_gamma, n_Op * n_gamma)
    end
    
    # Main scan loop
    for i_Op in 1:n_Op
        Threads.@threads for i_g in 1:n_gamma
            omega = CT(m * Omega_p_grid[i_Op], gamma_grid[i_g])
            det_grid[i_g, i_Op] = MatrixBuilder.compute_determinant(omega, precomputed, backend)
        end
        
        if verbose && mod(i_Op, 5) == 0
            print(".")
        end
    end
    
    if verbose
        println(" done.")
    end
    
    # Find minimum |det|
    Omega_p_init, gamma_init = find_minimum(det_grid, Omega_p_grid, gamma_grid)
    
    if verbose
        min_val = minimum(abs.(det_grid))
        @printf("Minimum |det| = %.2e at Ω_p = %.4f, γ = %.4f\n", min_val, Omega_p_init, gamma_init)
    end
    
    return det_grid, Omega_p_init, gamma_init
end

"""
    find_minimum(det_grid, Omega_p_grid, gamma_grid) -> (Omega_p, gamma)

Find the location of minimum |det| in the grid.
"""
function find_minimum(det_grid::Matrix{Complex{T}}, 
                      Omega_p_grid::Vector{T}, 
                      gamma_grid::Vector{T}) where T
    
    abs_det = abs.(det_grid)
    min_val, min_idx = findmin(abs_det)
    i_g_min, i_Op_min = Tuple(min_idx)
    
    Omega_p_init = Omega_p_grid[i_Op_min]
    gamma_init = gamma_grid[i_g_min]
    
    return Omega_p_init, gamma_init
end

"""
    grid_scan_distributed(config, precomputed, backend, gpu_devices; verbose=false)

Perform grid scan with work distributed across multiple GPUs.
"""
function grid_scan_distributed(config::KalnajsConfig, precomputed::PrecomputedData{T},
                               backend::GPUBackend.GPUBackendType,
                               gpu_devices::Vector{Int};
                               verbose::Bool=false) where T
    
    m = config.physics.m
    n_Op = config.scan.n_Op
    n_gamma = config.scan.n_gamma
    Omega_p_min = T(config.scan.Omega_p_min)
    Omega_p_max = T(config.scan.Omega_p_max)
    gamma_min = T(config.scan.gamma_min)
    gamma_max = T(config.scan.gamma_max)
    
    # Create grids
    Omega_p_grid = T.(range(Omega_p_min, Omega_p_max, length=n_Op))
    gamma_grid = T.(range(gamma_min, gamma_max, length=n_gamma))
    
    # Determinant grid
    CT = Complex{T}
    det_grid = zeros(CT, n_gamma, n_Op)
    
    # Total ω points
    total_points = n_Op * n_gamma
    n_gpus = length(gpu_devices)
    points_per_gpu = div(total_points, n_gpus)
    
    if verbose
        @printf("Distributed scan: %d GPUs, %d points total\n", n_gpus, total_points)
    end
    
    # Distribute work - each GPU processes a chunk of Omega_p values
    Op_per_gpu = div(n_Op, n_gpus)
    remainder = n_Op % n_gpus
    
    # Launch tasks for each GPU
    tasks = Task[]
    Op_start = 1
    
    for (i, device) in enumerate(gpu_devices)
        Op_count = Op_per_gpu + (i <= remainder ? 1 : 0)
        Op_end = Op_start + Op_count - 1
        Op_range = Op_start:Op_end
        
        task = Threads.@spawn begin
            GPUBackend.set_device!(backend, device)
            
            local_det = zeros(CT, n_gamma, Op_count)
            for (local_i, i_Op) in enumerate(Op_range)
                for i_g in 1:n_gamma
                    omega = CT(m * Omega_p_grid[i_Op], gamma_grid[i_g])
                    local_det[i_g, local_i] = MatrixBuilder.compute_determinant(omega, precomputed, backend)
                end
            end
            return (Op_range, local_det)
        end
        
        push!(tasks, task)
        Op_start = Op_end + 1
    end
    
    # Collect results
    for task in tasks
        Op_range, local_det = fetch(task)
        det_grid[:, Op_range] = local_det
    end
    
    # Find minimum
    Omega_p_init, gamma_init = find_minimum(det_grid, Omega_p_grid, gamma_grid)
    
    if verbose
        min_val = minimum(abs.(det_grid))
        @printf("Minimum |det| = %.2e at Ω_p = %.4f, γ = %.4f\n", min_val, Omega_p_init, gamma_init)
    end
    
    return det_grid, Omega_p_init, gamma_init
end

end # module GridScan
