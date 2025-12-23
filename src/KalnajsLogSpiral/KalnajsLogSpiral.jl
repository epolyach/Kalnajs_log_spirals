# src/KalnajsLogSpiral/KalnajsLogSpiral.jl
"""
Kalnajs Log-Spiral Eigenvalue Solver

Julia implementation of the nonlinear eigenvalue method for galactic normal modes
using logarithmic-spiral expansion (Kalnajs 1965, 1971; Zang 1976).

Features:
- Vendor-agnostic GPU acceleration (NVIDIA CUDA, AMD ROCm, Apple Metal)
- Multi-GPU support for parallel computation
- Adaptive parameter exploration for 6-8 digit precision
- Float32 default precision for GPU efficiency

Reference values for Toomre-Zang n=4, m=2 disk:
  - Zang (1976):              Ω_p = 0.439'426,      γ = 0.127'181
  - Polyachenko (refined):    Ω_p = 0.439'442'9284, γ = 0.127'204'5628
"""
module KalnajsLogSpiral

const VERSION = v"0.1.0"

# Core dependencies
using LinearAlgebra
using Printf
using SpecialFunctions
using TOML

# Include submodules in dependency order
include("GPUBackend.jl")
using .GPUBackend

include("Configuration.jl")
using .Configuration

include("Models.jl")
using .Models

include("OrbitIntegration.jl")
using .OrbitIntegration

include("BasisFunctions.jl")
using .BasisFunctions

include("MatrixBuilder.jl")
using .MatrixBuilder

include("GridScan.jl")
using .GridScan

include("NewtonSolver.jl")
using .NewtonSolver

# ============================================================================
# Public API
# ============================================================================

# Configuration
export KalnajsConfig, GridScanConfig, NewtonConfig, load_config

# GPU Backend
export get_available_backends, get_backend, to_gpu_array, synchronize_backend

# Models
export ToomreZangModel, create_toomre_zang_model

# Precomputation
export PrecomputedData, precompute_all

# Grid Scan
export grid_scan, find_minimum

# Newton Solver
export newton_search, convergence_study

# Main workflow
export run_eigenvalue_search

# ============================================================================
# Main Workflow Function
# ============================================================================

"""
    run_eigenvalue_search(config_file::String; kwargs...)
    run_eigenvalue_search(config::KalnajsConfig; kwargs...)

Run the complete eigenvalue search pipeline:
1. Precompute ω-independent quantities (orbits, W_l basis, N_m kernel)
2. Grid scan to find approximate zero location
3. Newton refinement to achieve target precision

# Arguments
- `config`: Configuration file path or KalnajsConfig object

# Keyword Arguments
- `verbose::Bool=true`: Print progress information
- `save_results::Bool=true`: Save results to file

# Returns
- `NamedTuple` with fields:
  - `Omega_p`: Pattern speed
  - `gamma`: Growth rate
  - `omega`: Complex frequency (m*Omega_p + i*gamma)
  - `det_val`: Final determinant value
  - `iterations`: Number of Newton iterations
  - `converged`: Whether convergence was achieved
"""
function run_eigenvalue_search(config_input::Union{String,KalnajsConfig};
                               verbose::Bool=true,
                               save_results::Bool=true)
    
    # Load configuration
    config = config_input isa String ? load_config(config_input) : config_input
    
    if verbose
        println("\n" * "="^60)
        println("KALNAJS LOG-SPIRAL EIGENVALUE SOLVER v$VERSION")
        println("="^60)
        print_config_summary(config)
    end
    
    # Initialize GPU backend
    backend = get_backend(config.gpu.backend)
    if verbose
        println("\n✓ GPU Backend: $(backend)")
    end
    
    # Step 1: Precompute ω-independent quantities
    if verbose
        println("\n--- Phase 1: Precomputation ---")
    end
    
    model = create_toomre_zang_model(config)
    precomputed = precompute_all(config, model, backend; verbose=verbose)
    
    # Step 2: Grid scan
    if verbose
        println("\n--- Phase 2: Grid Scan ---")
    end
    
    det_grid, Omega_p_init, gamma_init = grid_scan(config, precomputed, backend; verbose=verbose)
    
    if verbose
        @printf("Initial guess: Ω_p = %.6f, γ = %.6f\n", Omega_p_init, gamma_init)
    end
    
    # Step 3: Newton refinement
    if verbose
        println("\n--- Phase 3: Newton Refinement ---")
    end
    
    result = newton_search(config, precomputed, Omega_p_init, gamma_init, backend; verbose=verbose)
    
    # Print final result
    if verbose
        println("\n" * "="^60)
        println("FINAL RESULT")
        println("="^60)
        @printf("Ω_p = %.10f\n", result.Omega_p)
        @printf("γ   = %.10f\n", result.gamma)
        @printf("|det| = %.2e\n", abs(result.det_val))
        println("Converged: $(result.converged) ($(result.iterations) iterations)")
    end
    
    # Save results
    if save_results
        save_result_to_file(config, result)
    end
    
    return result
end

"""
Print configuration summary
"""
function print_config_summary(config::KalnajsConfig)
    println("\nConfiguration:")
    println("  Model: Toomre-Zang n=$(config.model.n_zang), m=$(config.physics.m)")
    println("  Grid: NR=$(config.grid.NR), Ne=$(config.grid.Ne), N_alpha=$(config.grid.N_alpha)")
    println("  Harmonics: l ∈ [$(config.grid.l_min), $(config.grid.l_max)]")
    println("  Precision: $(config.gpu.precision_double ? "Float64" : "Float32")")
    println("  GPU Backend: $(config.gpu.backend)")
end

"""
Save result to file
"""
function save_result_to_file(config::KalnajsConfig, result)
    # Create output directory if needed
    output_dir = get(config.io, :output_path, "results")
    mkpath(output_dir)
    
    # Generate filename with timestamp
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    filename = joinpath(output_dir, "eigenvalue_$(timestamp).toml")
    
    # Write result
    open(filename, "w") do io
        println(io, "# Kalnajs Log-Spiral Eigenvalue Result")
        println(io, "# Generated: $(Dates.now())")
        println(io)
        println(io, "[result]")
        @printf(io, "Omega_p = %.12f\n", result.Omega_p)
        @printf(io, "gamma = %.12f\n", result.gamma)
        @printf(io, "det_real = %.6e\n", real(result.det_val))
        @printf(io, "det_imag = %.6e\n", imag(result.det_val))
        println(io, "converged = $(result.converged)")
        println(io, "iterations = $(result.iterations)")
    end
end

end # module KalnajsLogSpiral
