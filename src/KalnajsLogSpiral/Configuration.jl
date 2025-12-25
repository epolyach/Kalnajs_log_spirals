# src/KalnajsLogSpiral/Configuration.jl
"""
Configuration management for Kalnajs Log-Spiral solver.

Supports TOML configuration files with the following sections:
- [physics]: Physical parameters (m, G)
- [model]: Toomre-Zang model parameters
- [grid]: Discretization parameters
- [scan]: Grid scan parameters
- [newton]: Newton refinement parameters
- [gpu]: GPU backend and precision settings
- [cpu]: CPU threading settings
- [io]: Input/output paths
"""
module Configuration

using TOML

export KalnajsConfig, PhysicsConfig, ModelConfig, GridConfig
export ScanConfig, NewtonConfig, GPUConfig, CPUConfig, IOConfig
export load_config, save_config
export get_float_type, get_complex_type, get_n_l, get_NPh

# ============================================================================
# Configuration Structs
# ============================================================================

"""
Physical parameters
"""
Base.@kwdef mutable struct PhysicsConfig
    m::Int = 2              # Azimuthal mode number
    G::Float64 = 1.0        # Gravitational constant
end

"""
Toomre-Zang model parameters
"""
Base.@kwdef mutable struct ModelConfig
    L0::Float64 = 1.0       # Angular momentum scale
    n_zang::Int = 4         # Zang taper exponent
    q1::Int = 7             # Radial velocity dispersion parameter (q_zang = q1 - 1)
end

"""
Grid discretization parameters
"""
Base.@kwdef mutable struct GridConfig
    # Radial and eccentricity grid
    NR::Int = 201           # Radial grid points
    Ne::Int = 21            # Eccentricity (v) grid points
    Rc_min::Float64 = 1e-3  # Minimum guiding center radius
    Rc_max::Float64 = 100.0 # Maximum guiding center radius
    v_min::Float64 = 1e-2   # Minimum circulation (v > 0 for prograde)
    
    # Alpha grid for log-spiral expansion
    N_alpha::Int = 101      # Alpha grid points
    alpha_max::Float64 = 10.0  # Maximum |α|
    
    # Radial harmonics
    l_min::Int = -25        # Minimum radial harmonic
    l_max::Int = 25         # Maximum radial harmonic
    
    # Orbit integration
    nw::Int = 10001         # Full orbit integration points
    nwa::Int = 201          # Reduced orbit grid for storage
end

"""
Grid scan parameters (fixed for locating approximate zero)
"""
Base.@kwdef mutable struct ScanConfig
    n_Op::Int = 41          # Ω_p grid points
    n_gamma::Int = 20       # γ grid points
    Omega_p_min::Float64 = 0.2   # Min pattern speed
    Omega_p_max::Float64 = 0.6   # Max pattern speed
    gamma_min::Float64 = 0.01    # Min growth rate
    gamma_max::Float64 = 0.2     # Max growth rate
end

"""
Newton refinement parameters (where parameter exploration happens)
"""
Base.@kwdef mutable struct NewtonConfig
    max_iter::Int = 50      # Maximum Newton iterations
    tol::Float64 = 1e-10    # Convergence tolerance
    delta::Float64 = 1e-6   # Finite difference step
    
    # Line search
    use_line_search::Bool = true
    max_line_search::Int = 10
    
    # Complex step derivative (more accurate than finite difference)
    use_complex_step::Bool = true
    complex_step_h::Float64 = 1e-20
    
    # Parameter exploration levels for convergence study
    # Each level increases resolution to verify convergence
    enable_convergence_study::Bool = false
end

"""
GPU backend and precision settings
"""
Base.@kwdef mutable struct GPUConfig
    backend::Symbol = :auto  # :auto, :CUDA, :AMDGPU, :Metal, :CPU
    precision_double::Bool = false  # false = Float32, true = Float64
    devices::Vector{Int} = Int[]    # GPU device IDs (empty = auto)
end

"""
CPU threading settings
"""
Base.@kwdef mutable struct CPUConfig
    precision_double::Bool = false  # false = Float32, true = Float64
    max_threads::Int = 32           # Maximum BLAS threads
end

"""
Input/output settings
"""
Base.@kwdef mutable struct IOConfig
    output_path::String = "results"
    save_intermediate::Bool = false
    verbose::Bool = true
end

"""
Complete configuration
"""
Base.@kwdef mutable struct KalnajsConfig
    physics::PhysicsConfig = PhysicsConfig()
    model::ModelConfig = ModelConfig()
    grid::GridConfig = GridConfig()
    scan::ScanConfig = ScanConfig()
    newton::NewtonConfig = NewtonConfig()
    gpu::GPUConfig = GPUConfig()
    cpu::CPUConfig = CPUConfig()
    io::IOConfig = IOConfig()
end

# ============================================================================
# Configuration Loading
# ============================================================================

"""
    load_config(filepath::String) -> KalnajsConfig

Load configuration from a TOML file.
"""
function load_config(filepath::String)
    if !isfile(filepath)
        error("Configuration file not found: $filepath")
    end
    
    data = TOML.parsefile(filepath)
    config = KalnajsConfig()
    
    # Physics section
    if haskey(data, "physics")
        p = data["physics"]
        haskey(p, "m") && (config.physics.m = p["m"])
        haskey(p, "G") && (config.physics.G = p["G"])
    end
    
    # Model section
    if haskey(data, "model")
        m = data["model"]
        haskey(m, "L0") && (config.model.L0 = m["L0"])
        haskey(m, "n_zang") && (config.model.n_zang = m["n_zang"])
        haskey(m, "q1") && (config.model.q1 = m["q1"])
    end
    
    # Grid section
    if haskey(data, "grid")
        g = data["grid"]
        haskey(g, "NR") && (config.grid.NR = g["NR"])
        haskey(g, "Ne") && (config.grid.Ne = g["Ne"])
        haskey(g, "Rc_min") && (config.grid.Rc_min = g["Rc_min"])
        haskey(g, "Rc_max") && (config.grid.Rc_max = g["Rc_max"])
        haskey(g, "v_min") && (config.grid.v_min = g["v_min"])
        haskey(g, "N_alpha") && (config.grid.N_alpha = g["N_alpha"])
        haskey(g, "alpha_max") && (config.grid.alpha_max = g["alpha_max"])
        haskey(g, "l_min") && (config.grid.l_min = g["l_min"])
        haskey(g, "l_max") && (config.grid.l_max = g["l_max"])
        haskey(g, "nw") && (config.grid.nw = g["nw"])
        haskey(g, "nwa") && (config.grid.nwa = g["nwa"])
    end
    
    # Scan section
    if haskey(data, "scan")
        s = data["scan"]
        haskey(s, "n_Op") && (config.scan.n_Op = s["n_Op"])
        haskey(s, "n_gamma") && (config.scan.n_gamma = s["n_gamma"])
        haskey(s, "Omega_p_min") && (config.scan.Omega_p_min = s["Omega_p_min"])
        haskey(s, "Omega_p_max") && (config.scan.Omega_p_max = s["Omega_p_max"])
        haskey(s, "gamma_min") && (config.scan.gamma_min = s["gamma_min"])
        haskey(s, "gamma_max") && (config.scan.gamma_max = s["gamma_max"])
    end
    
    # Newton section
    if haskey(data, "newton")
        n = data["newton"]
        haskey(n, "max_iter") && (config.newton.max_iter = n["max_iter"])
        haskey(n, "tol") && (config.newton.tol = n["tol"])
        haskey(n, "delta") && (config.newton.delta = n["delta"])
        haskey(n, "use_line_search") && (config.newton.use_line_search = n["use_line_search"])
        haskey(n, "max_line_search") && (config.newton.max_line_search = n["max_line_search"])
        haskey(n, "use_complex_step") && (config.newton.use_complex_step = n["use_complex_step"])
        haskey(n, "complex_step_h") && (config.newton.complex_step_h = n["complex_step_h"])
        haskey(n, "enable_convergence_study") && (config.newton.enable_convergence_study = n["enable_convergence_study"])
    end
    
    # Precision section (alternative to gpu/cpu sections)
    if haskey(data, "precision")
        prec = data["precision"]
        haskey(prec, "gpu_double_precision") && (config.gpu.precision_double = prec["gpu_double_precision"])
        haskey(prec, "cpu_double_precision") && (config.cpu.precision_double = prec["cpu_double_precision"])
    end
    
    # Precision section (alternative to gpu/cpu sections)
    if haskey(data, "precision")
        prec = data["precision"]
        haskey(prec, "gpu_double_precision") && (config.gpu.precision_double = prec["gpu_double_precision"])
        haskey(prec, "cpu_double_precision") && (config.cpu.precision_double = prec["cpu_double_precision"])
    end
    
    # GPU section
    if haskey(data, "gpu")
        gpu = data["gpu"]
        haskey(gpu, "backend") && (config.gpu.backend = Symbol(gpu["backend"]))
        haskey(gpu, "precision_double") && (config.gpu.precision_double = gpu["precision_double"])
        haskey(gpu, "devices") && (config.gpu.devices = gpu["devices"])
    end
    
    # CPU section
    if haskey(data, "cpu")
        cpu = data["cpu"]
        haskey(cpu, "precision_double") && (config.cpu.precision_double = cpu["precision_double"])
        haskey(cpu, "max_threads") && (config.cpu.max_threads = cpu["max_threads"])
    end
    
    # IO section
    if haskey(data, "io")
        io = data["io"]
        haskey(io, "output_path") && (config.io.output_path = io["output_path"])
        haskey(io, "save_intermediate") && (config.io.save_intermediate = io["save_intermediate"])
        haskey(io, "verbose") && (config.io.verbose = io["verbose"])
    end
    
    return config
end

"""
    save_config(config::KalnajsConfig, filepath::String)

Save configuration to a TOML file.
"""
function save_config(config::KalnajsConfig, filepath::String)
    open(filepath, "w") do io
        println(io, "# Kalnajs Log-Spiral Configuration")
        println(io)
        
        println(io, "[physics]")
        println(io, "m = $(config.physics.m)")
        println(io, "G = $(config.physics.G)")
        println(io)
        
        println(io, "[model]")
        println(io, "L0 = $(config.model.L0)")
        println(io, "n_zang = $(config.model.n_zang)")
        println(io, "q1 = $(config.model.q1)")
        println(io)
        
        println(io, "[grid]")
        println(io, "NR = $(config.grid.NR)")
        println(io, "Ne = $(config.grid.Ne)")
        println(io, "Rc_min = $(config.grid.Rc_min)")
        println(io, "Rc_max = $(config.grid.Rc_max)")
        println(io, "v_min = $(config.grid.v_min)")
        println(io, "N_alpha = $(config.grid.N_alpha)")
        println(io, "alpha_max = $(config.grid.alpha_max)")
        println(io, "l_min = $(config.grid.l_min)")
        println(io, "l_max = $(config.grid.l_max)")
        println(io, "nw = $(config.grid.nw)")
        println(io, "nwa = $(config.grid.nwa)")
        println(io)
        
        println(io, "[scan]")
        println(io, "n_Op = $(config.scan.n_Op)")
        println(io, "n_gamma = $(config.scan.n_gamma)")
        println(io, "Omega_p_min = $(config.scan.Omega_p_min)")
        println(io, "Omega_p_max = $(config.scan.Omega_p_max)")
        println(io, "gamma_min = $(config.scan.gamma_min)")
        println(io, "gamma_max = $(config.scan.gamma_max)")
        println(io)
        
        println(io, "[newton]")
        println(io, "max_iter = $(config.newton.max_iter)")
        println(io, "tol = $(config.newton.tol)")
        println(io, "delta = $(config.newton.delta)")
        println(io, "use_line_search = $(config.newton.use_line_search)")
        println(io, "use_complex_step = $(config.newton.use_complex_step)")
        println(io, "enable_convergence_study = $(config.newton.enable_convergence_study)")
        println(io)
        
        println(io, "[gpu]")
        println(io, "backend = \"$(config.gpu.backend)\"")
        println(io, "precision_double = $(config.gpu.precision_double)")
        println(io, "devices = $(config.gpu.devices)")
        println(io)
        
        println(io, "[cpu]")
        println(io, "precision_double = $(config.cpu.precision_double)")
        println(io, "max_threads = $(config.cpu.max_threads)")
        println(io)
        
        println(io, "[io]")
        println(io, "output_path = \"$(config.io.output_path)\"")
        println(io, "save_intermediate = $(config.io.save_intermediate)")
        println(io, "verbose = $(config.io.verbose)")
    end
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
Get the floating point type based on precision settings
"""
function get_float_type(config::KalnajsConfig, for_gpu::Bool=true)
    if for_gpu
        return config.gpu.precision_double ? Float64 : Float32
    else
        return config.cpu.precision_double ? Float64 : Float32
    end
end

"""
Get the complex floating point type based on precision settings
"""
function get_complex_type(config::KalnajsConfig, for_gpu::Bool=true)
    if for_gpu
        return config.gpu.precision_double ? ComplexF64 : ComplexF32
    else
        return config.cpu.precision_double ? ComplexF64 : ComplexF32
    end
end

"""
Get number of radial harmonics
"""
function get_n_l(config::KalnajsConfig)
    return config.grid.l_max - config.grid.l_min + 1
end

"""
Get total number of phase space points
"""
function get_NPh(config::KalnajsConfig)
    return config.grid.NR * config.grid.Ne
end


# Helper aliases
precision_gpu(config::KalnajsConfig) = get_float_type(config, true)
precision_cpu(config::KalnajsConfig) = get_float_type(config, false)
end # module Configuration
