# Kalnajs GPU Refactor Status

## Completed Work

1. ✅ Copied PME GPU backend (`GPUBackend.jl`) with NVIDIA/AMD support
2. ✅ Copied and adapted PME Toomre model (`Models.jl`) 
3. ✅ Implemented three-tier precision control:
   - Pre-GPU: Always Float64 (not critical for performance)
   - GPU: Configurable Float32/64 via `gpu.precision_double`
   - Post-GPU: Configurable Float32/64 via `cpu.precision_double`
4. ✅ Created `DevicePrecomputed` struct for GPU-resident data
5. ✅ Implemented GPU matrix assembly in `MatrixBuilder.jl`
   - CPU path: builds M on CPU
   - GPU path: builds M on device, transfers to host for det()
6. ✅ Simplified `GridScan.jl` and `NewtonSolver.jl` to route through unified determinant interface
7. ✅ Created PME-style `run_kalnajs_gpu.jl` with:
   - Auto GPU detection
   - CLI: `--gpu=auto|0|1|CPU`, `--threads=N`
   - Logging and device info
8. ✅ Removed old `run_kalnajs.jl` (single runner as requested)
9. ✅ Updated README with GPU usage documentation

## Current Issue

**Module compilation error**: `PrecomputedData` not defined in parent scope

### Problem
- Line 474 of `BasisFunctions.jl` (the `to_device` function) cannot resolve `PrecomputedData`
- Julia is looking for it in `KalnajsLogSpiral` module instead of `BasisFunctions` module
- This appears to be a module scoping issue

### Likely Cause
The `DevicePrecomputed` struct and `to_device` function were added at the end of `BasisFunctions.jl`. There may be a subtle scoping issue with how Julia resolves types in function signatures at module compilation time.

### Quick Fix Options

**Option 1**: Move Device* code earlier in file  
Move `DevicePrecomputed` struct and `to_device` function to right after `PrecomputedData` definition (after line 60).

**Option 2**: Explicit module prefix
Change line 474 from:
```julia
function to_device(precomp::PrecomputedData{T}, gpu_type, Tgpu::Type) where T
```
to:
```julia
function to_device(precomp::BasisFunctions.PrecomputedData{T}, gpu_type, Tgpu::Type) where T
```

**Option 3**: Split into separate module
Create `src/KalnajsLogSpiral/DeviceArrays.jl` that imports BasisFunctions and defines DevicePrecomputed/to_device there.

## Architecture Summary

The refactored code follows PME's pattern:

```
Input → Precompute (CPU, Float64)
     → Transfer to GPU (Tgpu)
     → Grid Scan (GPU matrix assembly, CPU det eval, output in Tcpu)
     → Newton (GPU matrix assembly, CPU det eval, in Tcpu)
     → Result (in Tcpu)
```

Key files:
- `run_kalnajs_gpu.jl`: Main runner (PME-style)
- `src/KalnajsLogSpiral/GPUBackend.jl`: GPU detection (from PME)
- `src/KalnajsLogSpiral/Models.jl`: Toomre physics (from PME)
- `src/KalnajsLogSpiral/MatrixBuilder.jl`: CPU/GPU determinant computation
- `src/KalnajsLogSpiral/BasisFunctions.jl`: Precomputation + device transfer (**needs fix**)

## Testing Plan (once fixed)

```bash
# Test CPU mode
julia --project=. run_kalnajs_gpu.jl configs/default.toml --gpu=CPU

# Test GPU 0
julia --project=. run_kalnajs_gpu.jl configs/default.toml --gpu=0

# Test GPU 1
julia --project=. run_kalnajs_gpu.jl configs/default.toml --gpu=1

# Test both GPUs (if multi-GPU support added)
julia --project=. run_kalnajs_gpu.jl configs/default.toml --gpu=01
```

## Hardware

Two NVIDIA GeForce RTX 2080 SUPER GPUs available.

