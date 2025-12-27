# src/KalnajsLogSpiral.jl
"""
Kalnajs Log-Spiral Eigenvalue Solver

Julia implementation for finding normal modes in Toomre-Zang galactic disks
using logarithmic-spiral expansion, with GPU acceleration (NVIDIA/AMD).
"""
module KalnajsLogSpiral

const VERSION = v"0.2.0"

# Load submodules in dependency order
include("KalnajsLogSpiral/GPUBackend.jl")
include("KalnajsLogSpiral/Configuration.jl")
include("KalnajsLogSpiral/Models.jl")
include("KalnajsLogSpiral/OrbitIntegration.jl")
include("KalnajsLogSpiral/BasisFunctions.jl")
include("KalnajsLogSpiral/MatrixBuilder.jl")
include("KalnajsLogSpiral/GridScan.jl")
include("KalnajsLogSpiral/NewtonSolver.jl")

# Re-export key types and functions
using .Configuration
export KalnajsConfig, load_config, load_model, save_config, ReferenceConfig
export get_float_type, get_complex_type, precision_gpu, precision_cpu

using .GPUBackend
export detect_gpu_type, get_gpu_type, gpu_device!, gpu_device_name, gpu_synchronize

using .Models
export ToomreZangModel, create_toomre_zang_model

using .BasisFunctions
export PrecomputedData, precompute_all, DevicePrecomputed, to_device

using .MatrixBuilder
export compute_determinant, compute_determinant_batched

using .GridScan
export grid_scan, find_minimum

using .NewtonSolver
export newton_search, NewtonResult

end # module KalnajsLogSpiral
