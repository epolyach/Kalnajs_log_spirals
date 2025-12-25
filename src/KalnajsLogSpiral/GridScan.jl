# src/KalnajsLogSpiral/GridScan.jl
"""
Grid scan to find approximate eigenvalue location.
Supports multi-GPU by distributing omega points across devices.
"""
module GridScan

using Printf
using ..Configuration
using ..BasisFunctions
using ..MatrixBuilder
using ..GPUBackend

export grid_scan, find_minimum

"""
    grid_scan(config, precomputed, devprecomp, gpu_type, Tcpu; 
              verbose=false, gpu_devices=Int[]) 
             -> (det_grid, Omega_p_init, gamma_init)

Perform grid scan over (Ω_p, γ) plane to locate approximate zero of det(I - M·dα).
If multiple gpu_devices are provided, distributes columns (Omega_p values) across GPUs.
Returns determinant grid and initial guesses.
"""
function grid_scan(config::KalnajsConfig, 
                  precomputed::PrecomputedData{T},
                  devprecomp::Union{Nothing,DevicePrecomputed},
                  gpu_type,
                  Tcpu::Type;
                  verbose::Bool=false,
                  gpu_devices::Vector{Int}=Int[]) where T
    
    # Grid parameters
    n_Op = config.scan.n_Op
    n_gamma = config.scan.n_gamma
    Op_min = config.scan.Omega_p_min
    Op_max = config.scan.Omega_p_max
    gamma_min = config.scan.gamma_min
    gamma_max = config.scan.gamma_max
    m = config.physics.m
    
    # Create grids
    Omega_p_grid = range(Tcpu(Op_min), Tcpu(Op_max), length=n_Op)
    gamma_grid = range(Tcpu(gamma_min), Tcpu(gamma_max), length=n_gamma)
    
    # Allocate output
    det_grid = Matrix{Complex{Tcpu}}(undef, n_gamma, n_Op)
    
    if verbose
        println("Scanning (Ω_p, γ) grid: $n_Op × $n_gamma = $(n_Op * n_gamma) points")
    end
    
    # Decide execution path
    n_gpus = length(gpu_devices)
    
    if n_gpus > 1 && gpu_type != GPUBackend.NONE && devprecomp !== nothing
        # Multi-GPU path
        if verbose
            println("Using $n_gpus GPUs: $gpu_devices")
        end
        scan_multi_gpu!(det_grid, config, precomputed, gpu_type, Tcpu, 
                       Omega_p_grid, gamma_grid, m, gpu_devices, verbose)
    elseif gpu_type != GPUBackend.NONE && devprecomp !== nothing
        # Single GPU path
        if verbose
            gpu_name = GPUBackend.gpu_device_name()
            println("Using single GPU: $gpu_name")
        end
        scan_single_gpu!(det_grid, precomputed, devprecomp, gpu_type, Tcpu,
                        Omega_p_grid, gamma_grid, m, verbose)
    else
        # CPU path
        if verbose
            println("Using CPU")
        end
        scan_cpu!(det_grid, precomputed, Tcpu, Omega_p_grid, gamma_grid, m, verbose)
    end
    
    # Find minimum
    Omega_p_init, gamma_init = find_minimum(det_grid, collect(Omega_p_grid), collect(gamma_grid))
    
    if verbose
        min_det = minimum(abs.(det_grid))
        @printf("Minimum |det| = %.2e at Ω_p = %.6f, γ = %.6f\n", min_det, Omega_p_init, gamma_init)
    end
    
    return det_grid, Omega_p_init, gamma_init
end

"""
Single GPU scan - straightforward loop.
"""
function scan_single_gpu!(det_grid, precomputed, devprecomp, gpu_type, Tcpu,
                         Omega_p_grid, gamma_grid, m, verbose)
    T = eltype(precomputed.d_alpha)
    n_Op = length(Omega_p_grid)
    n_gamma = length(gamma_grid)
    total_points = n_Op * n_gamma
    completed = 0
    
    for (j, Op) in enumerate(Omega_p_grid)
        for (i, gamma) in enumerate(gamma_grid)
            omega = Complex{T}(m * Op, gamma)
            det_grid[i, j] = MatrixBuilder.compute_determinant(
                omega, precomputed, devprecomp, gpu_type, Tcpu
            )
            
            completed += 1
            if verbose && completed % 100 == 0
                progress = 100 * completed / total_points
                @printf("\rProgress: %.1f%%", progress)
            end
        end
    end
    
    if verbose
        println("\rProgress: 100.0%")
    end
end

"""
Multi-GPU scan - distribute Omega_p columns across GPUs.
Each GPU processes a subset of columns independently.
"""
function scan_multi_gpu!(det_grid, config, precomputed, gpu_type, Tcpu,
                        Omega_p_grid, gamma_grid, m, gpu_devices, verbose)
    T = eltype(precomputed.d_alpha)
    n_Op = length(Omega_p_grid)
    n_gamma = length(gamma_grid)
    n_gpus = length(gpu_devices)
    
    # Distribute columns across GPUs
    cols_per_gpu = div(n_Op, n_gpus)
    remainder = n_Op % n_gpus
    
    # Create column ranges for each GPU
    col_ranges = Vector{UnitRange{Int}}(undef, n_gpus)
    start_col = 1
    for g in 1:n_gpus
        extra = g <= remainder ? 1 : 0
        end_col = start_col + cols_per_gpu + extra - 1
        col_ranges[g] = start_col:end_col
        start_col = end_col + 1
    end
    
    if verbose
        for (g, dev_id) in enumerate(gpu_devices)
            println("  GPU $dev_id: columns $(col_ranges[g]) ($(length(col_ranges[g])) cols)")
        end
    end
    
    # Get GPU precision type
    Tgpu = config.gpu.precision_double ? Float64 : Float32
    
    # Launch tasks for each GPU
    tasks = Vector{Task}(undef, n_gpus)
    results = Vector{Matrix{Complex{Tcpu}}}(undef, n_gpus)
    
    for (g, dev_id) in enumerate(gpu_devices)
        col_range = col_ranges[g]
        tasks[g] = Threads.@spawn begin
            # Set GPU device
            GPUBackend.gpu_device!(dev_id)
            
            # Create device precomputed data for this GPU
            devprecomp_local = BasisFunctions.to_device(precomputed, gpu_type, Tgpu)
            
            # Local result matrix
            n_cols = length(col_range)
            local_det = Matrix{Complex{Tcpu}}(undef, n_gamma, n_cols)
            
            # Scan assigned columns
            for (local_j, j) in enumerate(col_range)
                Op = Omega_p_grid[j]
                for (i, gamma) in enumerate(gamma_grid)
                    omega = Complex{T}(m * Op, gamma)
                    local_det[i, local_j] = MatrixBuilder.compute_determinant(
                        omega, precomputed, devprecomp_local, gpu_type, Tcpu
                    )
                end
            end
            
            GPUBackend.gpu_synchronize()
            local_det
        end
    end
    
    # Collect results
    for (g, task) in enumerate(tasks)
        results[g] = fetch(task)
        det_grid[:, col_ranges[g]] .= results[g]
    end
    
    if verbose
        println("Progress: 100.0%")
    end
end

"""
CPU scan path.
"""
function scan_cpu!(det_grid, precomputed, Tcpu, Omega_p_grid, gamma_grid, m, verbose)
    T = eltype(precomputed.d_alpha)
    n_Op = length(Omega_p_grid)
    n_gamma = length(gamma_grid)
    total_points = n_Op * n_gamma
    completed = 0
    
    for (j, Op) in enumerate(Omega_p_grid)
        for (i, gamma) in enumerate(gamma_grid)
            omega = Complex{T}(m * Op, gamma)
            det_grid[i, j] = MatrixBuilder.compute_determinant_cpu(
                omega, precomputed, Tcpu
            )
            
            completed += 1
            if verbose && completed % 100 == 0
                progress = 100 * completed / total_points
                @printf("\rProgress: %.1f%%", progress)
            end
        end
    end
    
    if verbose
        println("\rProgress: 100.0%")
    end
end

"""
    find_minimum(det_grid, Omega_p_grid, gamma_grid) -> (Omega_p_init, gamma_init)

Find location of minimum |det| in the grid.
"""
function find_minimum(det_grid::Matrix{Complex{T}}, 
                     Omega_p_grid::Vector{T}, 
                     gamma_grid::Vector{T}) where T
    
    abs_det = abs.(det_grid)
    min_idx = argmin(abs_det)
    i_gamma, i_Op = Tuple(min_idx)
    
    Omega_p_init = Omega_p_grid[i_Op]
    gamma_init = gamma_grid[i_gamma]
    
    return Omega_p_init, gamma_init
end

end # module GridScan
