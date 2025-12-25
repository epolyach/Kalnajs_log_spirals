# Kalnajs Log Spirals

This project explores normal modes in galactic disks using logarithmic-spiral expansion methods, following the approaches of Kalnajs (1965, 1971) and Zang (1976).

## Overview

The code implements matrix methods for computing non-axisymmetric normal modes in hot stellar disks, focusing on:
- Logarithmic-exponential basis functions for mode decomposition
- Toomre-Zang disk models (n=4 cut-out disks)
- Nonlinear eigenvalue formulations
- Newton-Raphson searches for unstable modes

## Quick Start

```bash
# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run with default configuration (auto-detect GPU)
julia --project=. run_kalnajs_gpu.jl --config=configs/default.toml --gpu=auto

# Run on specific GPUs (e.g., devices 0 and 1)
julia --project=. run_kalnajs_gpu.jl --config=configs/default.toml --gpu=01

# Run on CPU only
julia --project=. run_kalnajs_gpu.jl --config=configs/default.toml --gpu=CPU

# With BLAS thread limit
julia --project=. run_kalnajs_gpu.jl --config=configs/default.toml --gpu=01 --threads=4
```

## Command-Line Options

```
Usage: julia --project=. run_kalnajs_gpu.jl [options]

Options:
  --config=FILE    Configuration file (default: configs/default.toml)
  --gpu=IDS        GPU device selection:
                     auto - auto-detect available GPUs
                     0, 1, 01 - specific device(s)
                     CPU - force CPU mode
  --threads=N      BLAS threads limit (default: from config)
  --help, -h       Show help
```

## Configuration

Configuration is managed via TOML files in `configs/`. Example (`configs/default.toml`):

```toml
[physics]
m = 2                     # Azimuthal mode number

[model]
n_zang = 4                # Zang cut-out parameter
q1 = 7                    # Toomre Q parameter
G = 1.0                   # Gravitational constant

[grid]
NR = 501                  # Radial grid points
Ne = 51                   # Energy grid points
N_alpha = 301             # Alpha integration points
alpha_max = 30.0          # Maximum alpha
l_min = -100              # Minimum l index
l_max = 100               # Maximum l index

[precision]
gpu_double_precision = true
cpu_double_precision = true

[newton]
max_iter = 50
tol = 1e-12
delta = 1e-6

[cpu]
max_threads = 4

[io]
output_path = "results"

[reference]
Omega_p = 0.4397142267    # Initial guess for pattern speed
gamma = 0.1230663944      # Initial guess for growth rate
```

## Project Structure

```
 run_kalnajs_gpu.jl          # Main GPU runner script
 configs/
   └── default.toml            # Default configuration
 src/
   ├── KalnajsLogSpiral.jl     # Main module
   └── KalnajsLogSpiral/
       ├── GPUBackend.jl       # Vendor-agnostic GPU abstraction
       ├── Configuration.jl    # TOML config handling
       ├── Models.jl           # Toomre-Zang model
       ├── OrbitIntegration.jl # Orbit calculations
       ├── BasisFunctions.jl   # W_l and N_m(α) kernel
       ├── MatrixBuilder.jl    # M(ω) matrix construction
       ├── GridScan.jl         # Complex ω-plane scanning
       └── NewtonSolver.jl     # Newton-Raphson solver
 Matlab/                     # Original MATLAB implementations
 Matrix_log_exp.pdf          # Mathematical documentation
 results/                    # Output directory (gitignored)
```

## Julia Implementation Features

- **Vendor-agnostic GPU acceleration** via KernelAbstractions.jl
  - NVIDIA GPUs (CUDA.jl)
  - AMD GPUs (AMDGPU.jl / ROCm)
  - CPU fallback
- **Multi-GPU support** for distributed computation
- **Configurable precision** (Float32/Float64)
- **Adaptive Newton refinement** for high precision

## Reference Values

For Toomre-Zang n=4, m=2 disk:

| Source | Ω_p | γ |
|--------|-----|---|
| Zang (1976) | 0.439'426 | 0.127'181 |
| Polyachenko (refined) | 0.439'442'9284 | 0.127'204'5628 |

## Standalone Julia Scripts

Direct translations of the MATLAB code:

```bash
# CPU version
julia --project=. NL_julia_direct.jl

# GPU version (CUDA)
julia --project=. NL_julia_direct_gpu.jl
```

## Documentation

See **`Matrix_log_exp.pdf`** for detailed mathematical formulation, derivations, and theoretical background.

## MATLAB Files

Original MATLAB implementations are preserved in the `Matlab/` directory for reference.

## References

- Kalnajs, A. J. 1965, PhD thesis, Harvard University
- Kalnajs, A. J. 1971, ApJ, 166, 275
- Zang, T. A. 1976, PhD thesis, MIT
- Toomre, A. 1981, in "Structure and Evolution of Normal Galaxies"

## Authors

E. V. Polyachenko, I. G. Shukhman  
December 2025
