#!/usr/bin/env julia

const HELP_TEXT = """
Kalnajs Log-Spiral GPU Runner - Modular Package Version

Usage: julia --project=. run_kalnajs_gpu.jl config.toml model.toml [options]

Arguments:
  config.toml    Configuration file path
  model.toml     Model file path

Options:
  --gpu=IDS        GPU device selection (auto, 0, 1, 01, CPU)
  --threads=N      BLAS threads limit (default: from config)

Examples:
  julia --project=. run_kalnajs_gpu.jl configs/default.toml models/kalnajs_zang4.toml --gpu=auto
  julia --project=. run_kalnajs_gpu.jl configs/large.toml models/kalnajs_zang4.toml --gpu=01 --threads=4
"""
using Pkg
Pkg.activate(".")

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using KalnajsLogSpiral
using TOML
using LinearAlgebra
using Printf
using Dates

# Include GPUBackend for detection
const GPU_TYPE = KalnajsLogSpiral.GPUBackend.detect_gpu_type()

if GPU_TYPE == KalnajsLogSpiral.GPUBackend.NVIDIA
    using CUDA
    println("🎮 Detected NVIDIA GPU - using CUDA.jl")
elseif GPU_TYPE == KalnajsLogSpiral.GPUBackend.AMD
    using AMDGPU
    println("🎮 Detected AMD GPU - using AMDGPU.jl")
else
    println("ℹ️  No GPU detected - will use CPU")
end

# Backend-agnostic GPU helper functions
function gpu_ndevices()
    if GPU_TYPE == KalnajsLogSpiral.GPUBackend.NVIDIA
        return CUDA.ndevices()
    elseif GPU_TYPE == KalnajsLogSpiral.GPUBackend.AMD
        return length(AMDGPU.devices())
    end
    return 0
end

function gpu_device!(id::Int)
    if GPU_TYPE == KalnajsLogSpiral.GPUBackend.NVIDIA
        CUDA.device!(id)
    elseif GPU_TYPE == KalnajsLogSpiral.GPUBackend.AMD
        AMDGPU.device!(AMDGPU.devices()[id + 1])
    end
end

function gpu_device_name()
    if GPU_TYPE == KalnajsLogSpiral.GPUBackend.NVIDIA
        return CUDA.name(CUDA.device())
    elseif GPU_TYPE == KalnajsLogSpiral.GPUBackend.AMD
        return string(AMDGPU.device())
    end
    return "No GPU"
end

function gpu_total_memory()
    if GPU_TYPE == KalnajsLogSpiral.GPUBackend.NVIDIA
        return CUDA.total_memory()
    elseif GPU_TYPE == KalnajsLogSpiral.GPUBackend.AMD
        return 24 * 1024^3
    end
    return 0
end

# Global log file
const LOG_FILE = Ref{Union{IOStream, Nothing}}(nothing)
const LOG_PATH = Ref{String}("")

function log_print(args...)
    print(args...)
    if LOG_FILE[] !== nothing
        print(LOG_FILE[], args...)
        flush(LOG_FILE[])
    end
end

function log_println(args...)
    println(args...)
    if LOG_FILE[] !== nothing
        println(LOG_FILE[], args...)
        flush(LOG_FILE[])
    end
end

function start_global_logging(config)
    log_dir = config.io.output_path
    mkpath(log_dir)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    log_filename = @sprintf("kalnajs_m%d_n%d_%s.log",
                           config.physics.m, config.model.n_zang, timestamp)
    LOG_PATH[] = joinpath(log_dir, log_filename)
    LOG_FILE[] = open(LOG_PATH[], "w")
end

function stop_global_logging()
    if LOG_FILE[] !== nothing
        flush(LOG_FILE[])
        close(LOG_FILE[])
        LOG_FILE[] = nothing
    end
end

# Parse arguments
function parse_arguments()
    if length(ARGS) < 2
        println("Error: Insufficient arguments")
        println(HELP_TEXT)
        exit(1)
    end
    
    if ARGS[1] == "--help" || ARGS[1] == "-h"
        println(HELP_TEXT)
        exit(0)
    end
    
    config_file = ARGS[1]
    model_file = ARGS[2]
    gpu_spec = "auto"
    threads = nothing
    
    for arg in ARGS[3:end]
        if startswith(arg, "--gpu=")
            gpu_spec = split(arg, "=")[2]
        elseif startswith(arg, "--threads=")
            threads = parse(Int, split(arg, "=")[2])
        elseif arg == "--help" || arg == "-h"
            println(HELP_TEXT)
            exit(0)
        end
    end
    
    return config_file, model_file, gpu_spec, threads
end

config_file, model_file, gpu_spec, threads_arg = parse_arguments()

if !isfile(config_file)
    println("Error: Configuration file not found: $config_file")
    exit(1)
end

if !isfile(model_file)
    println("Error: Model file not found: $model_file")
    exit(1)
end

config = KalnajsLogSpiral.load_config(config_file)
config = KalnajsLogSpiral.load_model(config, model_file)
start_global_logging(config)

log_println()
log_println("="^60)
log_println("KALNAJS LOG-SPIRAL EIGENVALUE SOLVER - GPU VERSION")
log_println("="^60)
log_println("Started: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
log_println("Config: $config_file")
log_println("Model:  $model_file")
log_println()

# GPU Setup
gpu_id = 0
gpu_devices = Int[]
use_gpu = true

if gpu_spec == "auto"
    if GPU_TYPE != KalnajsLogSpiral.GPUBackend.NONE
        n_gpus = gpu_ndevices()
        gpu_devices = collect(0:n_gpus-1)
        gpu_id = 0
        log_println("GPU:    Auto-detected $n_gpus device(s)")
    else
        use_gpu = false
        log_println("GPU:    None (using CPU)")
    end
elseif gpu_spec == "CPU" || gpu_spec == "cpu"
    use_gpu = false
    log_println("GPU:    CPU mode (forced)")
elseif all(isdigit, gpu_spec)
    if length(gpu_spec) == 1
        gpu_id = parse(Int, gpu_spec)
        gpu_devices = [gpu_id]
    else
        gpu_devices = [parse(Int, string(c)) for c in gpu_spec]
        gpu_id = gpu_devices[1]
    end
    log_println("GPU:    Device(s) $gpu_devices")
else
    println("Error: Invalid GPU specification: $gpu_spec")
    exit(1)
end

# Validate GPU devices
if use_gpu && GPU_TYPE != KalnajsLogSpiral.GPUBackend.NONE
    n_available = gpu_ndevices()
    for dev_id in gpu_devices
        if dev_id >= n_available
            log_println("Error: GPU $dev_id not available (have $n_available GPUs)")
            stop_global_logging()
            exit(1)
        end
    end
    
    gpu_device!(gpu_id)
    local dev_name = gpu_device_name()
    local mem_gb = gpu_total_memory() / 1024^3
    local backend_str = GPU_TYPE == KalnajsLogSpiral.GPUBackend.NVIDIA ? "CUDA" : "ROCm"
    
    if length(gpu_devices) == 1
        log_println(@sprintf("        Device %d: %s (%.1f GB) [%s]", 
                            gpu_id, dev_name, mem_gb, backend_str))
    else
        log_println("        Multi-GPU mode:")
        for dev_id in gpu_devices
            gpu_device!(dev_id)
            local name = gpu_device_name()
            local mem = gpu_total_memory() / 1024^3
            log_println(@sprintf("        - Device %d: %s (%.1f GB)", dev_id, name, mem))
        end
        gpu_device!(gpu_id)
    end
end

# BLAS threads
blas_threads = something(threads_arg, config.cpu.max_threads)
blas_threads = min(blas_threads, config.cpu.max_threads, Threads.nthreads())
BLAS.set_num_threads(blas_threads)
log_println("Threads: $blas_threads BLAS threads")
log_println()

# Precision settings
Tgpu = KalnajsLogSpiral.Configuration.get_float_type(config, true)
Tcpu = KalnajsLogSpiral.Configuration.get_float_type(config, false)
log_println("Precision:")
log_println("  GPU: ", use_gpu ? Tgpu : "N/A")
log_println("  CPU: ", Tcpu)
log_println()

# Configuration summary
log_println("="^60)
log_println("CONFIGURATION")
log_println("="^60)
log_println("Physics: m = ", config.physics.m)
log_println("Model: n_zang = ", config.model.n_zang, ", q1 = ", config.model.q1)
log_println("Grid: NR×Ne = ", config.grid.NR, "×", config.grid.Ne, ", N_alpha = ", config.grid.N_alpha)
log_println("Newton: max_iter = ", config.newton.max_iter, ", tol = ", config.newton.tol)
log_println()

# Reference values
log_println(@sprintf("Reference: Ω_p = %.10f, γ = %.10f", config.reference.Omega_p, config.reference.gamma))
log_println()

start_time = time()

try
    # Phase 1: Precomputation (always Float64, CPU)
    log_println("="^60)
    log_println("PHASE 1: PRECOMPUTATION (Float64, CPU)")
    log_println("="^60)
    
    model = KalnajsLogSpiral.create_toomre_zang_model(config)
    precomputed = KalnajsLogSpiral.precompute_all(config, model, GPU_TYPE; verbose=true)
    
    log_println()
    
    # Transfer to GPU if needed
    devprecomp = nothing
    if use_gpu && GPU_TYPE != KalnajsLogSpiral.GPUBackend.NONE
        log_println("Transferring arrays to GPU in precision $Tgpu...")
        devprecomp = KalnajsLogSpiral.to_device(precomputed, GPU_TYPE, Tgpu)
        log_println("✓ GPU transfer complete")
        log_println()
    end
    
    # Phase 2: Grid Scan (DISABLED - using reference values)
    log_println("="^60)
    log_println("PHASE 2: GRID SCAN (SKIPPED)")
    log_println("="^60)
    
    # Use reference values as initial guess
    Omega_p_init = Tcpu(config.reference.Omega_p)
    gamma_init = Tcpu(config.reference.gamma)
    
    log_println()
    log_println(@sprintf("Using reference initial guess: Ω_p = %.8f, γ = %.8f", Omega_p_init, gamma_init))
    log_println()
    
    # Phase 3: Newton Refinement
    log_println("="^60)
    log_println("PHASE 3: NEWTON REFINEMENT")
    log_println("="^60)
    
    result = KalnajsLogSpiral.newton_search(
        config, precomputed, devprecomp, GPU_TYPE, Tcpu,
        Omega_p_init, gamma_init; verbose=true
    )
    
    # Final results
    elapsed = time() - start_time
    minutes = floor(Int, elapsed / 60)
    seconds = elapsed - 60 * minutes
    
    log_println()
    log_println("="^60)
    log_println("FINAL RESULT")
    log_println("="^60)
    log_println(@sprintf("Ω_p = %.12f", result.Omega_p))
    log_println(@sprintf("γ   = %.12f", result.gamma))
    log_println(@sprintf("|det| = %.2e", abs(result.det_val)))
    log_println("Converged: ", result.converged, " (", result.iterations, " iterations)")
    log_println()
    log_println("Runtime: ", @sprintf("%.1fs (%d min %.1fs)", elapsed, minutes, seconds))
    log_println("="^60)
    
catch e
    elapsed = time() - start_time
    minutes = floor(Int, elapsed / 60)
    
    log_println()
    log_println("="^60)
    log_println("ERROR")
    log_println("="^60)
    log_println("Error after $minutes minutes:")
    log_println("Type: ", typeof(e))
    log_println("Message: ", sprint(showerror, e))
    for (exc, bt) in current_exceptions()
        showerror(stdout, exc, bt)
        println()
    end
    log_println()
    log_println("Log file: ", LOG_PATH[])
    log_println("="^60)
    
    stop_global_logging()
    rethrow(e)
end

stop_global_logging()
println("Log saved to: ", LOG_PATH[])
