#!/usr/bin/env julia

"""
Kalnajs Log-Spiral Eigenvalue Solver

Julia implementation for finding normal modes in Toomre-Zang galactic disks
using logarithmic-spiral expansion.

Usage:
    julia --project=. run_kalnajs.jl [config.toml] [--gpu=auto|CUDA|AMDGPU|Metal|CPU]

Examples:
    julia --project=. run_kalnajs.jl
    julia --project=. run_kalnajs.jl configs/default.toml
    julia --project=. run_kalnajs.jl configs/default.toml --gpu=CPU
    julia --threads=4 --project=. run_kalnajs.jl configs/default.toml

Reference values for Toomre-Zang n=4, m=2 disk:
  - Zang (1976):              Ω_p = 0.439'426,      γ = 0.127'181
  - Polyachenko (refined):    Ω_p = 0.439'442'9284, γ = 0.127'204'5628
"""

using Pkg
Pkg.activate(".")

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using KalnajsLogSpiral

function parse_arguments()
    config_file = "configs/default.toml"
    gpu_backend = :auto
    
    for arg in ARGS
        if endswith(arg, ".toml")
            config_file = arg
        elseif startswith(arg, "--gpu=")
            backend_str = split(arg, "=")[2]
            gpu_backend = Symbol(backend_str)
        end
    end
    
    return config_file, gpu_backend
end

function main()
    config_file, gpu_backend = parse_arguments()
    
    # Check config file exists
    if !isfile(config_file)
        println("Error: Configuration file not found: $config_file")
        println("Usage: julia --project=. run_kalnajs.jl [config.toml] [--gpu=...]")
        exit(1)
    end
    
    # Load configuration
    config = load_config(config_file)
    
    # Override GPU backend if specified
    if gpu_backend != :auto
        config.gpu.backend = gpu_backend
    end
    
    # Run eigenvalue search
    result = run_eigenvalue_search(config; verbose=true, save_results=true)
    
    # Print final result with formatted output
    println("\n" * "="^60)
    println("REFERENCE VALUES FOR COMPARISON")
    println("="^60)
    println("Zang (1976):           Ω_p = 0.439'426      γ = 0.127'181")
    println("Polyachenko (refined): Ω_p = 0.439'442'9284 γ = 0.127'204'5628")
    println()
    
    return result
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
