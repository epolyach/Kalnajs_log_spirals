# src/KalnajsLogSpiral/GPUBackend.jl
"""
Vendor-agnostic GPU backend abstraction using KernelAbstractions.jl

Supports:
- NVIDIA GPUs via CUDA.jl
- AMD GPUs via AMDGPU.jl
- Apple GPUs via Metal.jl
- CPU fallback

The same kernels work across all backends without modification.
"""
module GPUBackend

using KernelAbstractions

export get_available_backends, get_backend, get_backend_type
export to_gpu_array, to_cpu_array, synchronize_backend
export get_device_count, set_device!, get_device_info
export GPUBackendType, CPUBackendType, CUDABackendType, AMDGPUBackendType, MetalBackendType

# ============================================================================
# Backend Type Definitions
# ============================================================================

abstract type GPUBackendType end
struct CPUBackendType <: GPUBackendType end
struct CUDABackendType <: GPUBackendType end
struct AMDGPUBackendType <: GPUBackendType end
struct MetalBackendType <: GPUBackendType end

# Global state for loaded backends
const _cuda_available = Ref{Union{Bool, Nothing}}(nothing)
const _amdgpu_available = Ref{Union{Bool, Nothing}}(nothing)
const _metal_available = Ref{Union{Bool, Nothing}}(nothing)

# ============================================================================
# Backend Detection
# ============================================================================

"""
    check_cuda_available() -> Bool

Check if CUDA.jl is available and functional.
"""
function check_cuda_available()
    if _cuda_available[] !== nothing
        return _cuda_available[]
    end
    
    try
        @eval begin
            using CUDA
            _cuda_available[] = CUDA.functional()
        end
    catch
        _cuda_available[] = false
    end
    
    return _cuda_available[]
end

"""
    check_amdgpu_available() -> Bool

Check if AMDGPU.jl is available and functional.
"""
function check_amdgpu_available()
    if _amdgpu_available[] !== nothing
        return _amdgpu_available[]
    end
    
    try
        @eval begin
            using AMDGPU
            _amdgpu_available[] = AMDGPU.functional()
        end
    catch
        _amdgpu_available[] = false
    end
    
    return _amdgpu_available[]
end

"""
    check_metal_available() -> Bool

Check if Metal.jl is available and functional.
"""
function check_metal_available()
    if _metal_available[] !== nothing
        return _metal_available[]
    end
    
    try
        @eval begin
            using Metal
            _metal_available[] = Metal.functional()
        end
    catch
        _metal_available[] = false
    end
    
    return _metal_available[]
end

"""
    get_available_backends() -> Vector{Symbol}

Return list of available GPU backends on this system.
"""
function get_available_backends()
    backends = [:CPU]
    
    if check_cuda_available()
        push!(backends, :CUDA)
    end
    
    if check_amdgpu_available()
        push!(backends, :AMDGPU)
    end
    
    if check_metal_available()
        push!(backends, :Metal)
    end
    
    return backends
end

"""
    get_backend(backend::Symbol=:auto) -> GPUBackendType

Get the specified backend, or auto-detect the best available.

# Arguments
- `backend`: One of `:auto`, `:CPU`, `:CUDA`, `:AMDGPU`, `:Metal`
"""
function get_backend(backend::Symbol=:auto)
    if backend == :auto
        # Priority: CUDA > AMDGPU > Metal > CPU
        if check_cuda_available()
            return CUDABackendType()
        elseif check_amdgpu_available()
            return AMDGPUBackendType()
        elseif check_metal_available()
            return MetalBackendType()
        else
            return CPUBackendType()
        end
    elseif backend == :CPU
        return CPUBackendType()
    elseif backend == :CUDA
        if !check_cuda_available()
            @warn "CUDA requested but not available, falling back to CPU"
            return CPUBackendType()
        end
        return CUDABackendType()
    elseif backend == :AMDGPU
        if !check_amdgpu_available()
            @warn "AMDGPU requested but not available, falling back to CPU"
            return CPUBackendType()
        end
        return AMDGPUBackendType()
    elseif backend == :Metal
        if !check_metal_available()
            @warn "Metal requested but not available, falling back to CPU"
            return CPUBackendType()
        end
        return MetalBackendType()
    else
        error("Unknown backend: $backend. Valid options: :auto, :CPU, :CUDA, :AMDGPU, :Metal")
    end
end

# ============================================================================
# Array Conversion
# ============================================================================

"""
    to_gpu_array(A::AbstractArray, backend::GPUBackendType, T::Type=eltype(A))

Convert array to GPU array for the specified backend.
"""
function to_gpu_array(A::AbstractArray, ::CPUBackendType, T::Type=eltype(A))
    return T == eltype(A) ? A : T.(A)
end

function to_gpu_array(A::AbstractArray, ::CUDABackendType, T::Type=eltype(A))
    @eval using CUDA
    return CuArray(T.(A))
end

function to_gpu_array(A::AbstractArray, ::AMDGPUBackendType, T::Type=eltype(A))
    @eval using AMDGPU
    return ROCArray(T.(A))
end

function to_gpu_array(A::AbstractArray, ::MetalBackendType, T::Type=eltype(A))
    @eval using Metal
    return MtlArray(T.(A))
end

"""
    to_cpu_array(A) -> Array

Convert GPU array back to CPU array.
"""
function to_cpu_array(A::AbstractArray)
    return Array(A)
end

# ============================================================================
# Synchronization
# ============================================================================

"""
    synchronize_backend(backend::GPUBackendType)

Synchronize the GPU backend (wait for all operations to complete).
"""
synchronize_backend(::CPUBackendType) = nothing

function synchronize_backend(::CUDABackendType)
    @eval using CUDA
    CUDA.synchronize()
end

function synchronize_backend(::AMDGPUBackendType)
    @eval using AMDGPU
    AMDGPU.synchronize()
end

function synchronize_backend(::MetalBackendType)
    @eval using Metal
    Metal.synchronize()
end

# ============================================================================
# Device Management
# ============================================================================

"""
    get_device_count(backend::GPUBackendType) -> Int

Get the number of available devices for this backend.
"""
get_device_count(::CPUBackendType) = 1

function get_device_count(::CUDABackendType)
    @eval using CUDA
    return CUDA.ndevices()
end

function get_device_count(::AMDGPUBackendType)
    @eval using AMDGPU
    return length(AMDGPU.devices())
end

function get_device_count(::MetalBackendType)
    @eval using Metal
    return length(Metal.devices())
end

"""
    set_device!(backend::GPUBackendType, device_id::Int)

Set the active device for this backend.
"""
set_device!(::CPUBackendType, ::Int) = nothing

function set_device!(::CUDABackendType, device_id::Int)
    @eval using CUDA
    CUDA.device!(device_id)
end

function set_device!(::AMDGPUBackendType, device_id::Int)
    @eval using AMDGPU
    AMDGPU.device!(AMDGPU.devices()[device_id + 1])
end

function set_device!(::MetalBackendType, device_id::Int)
    @eval using Metal
    Metal.device!(Metal.devices()[device_id + 1])
end

"""
    get_device_info(backend::GPUBackendType) -> NamedTuple

Get information about the current device.
"""
function get_device_info(::CPUBackendType)
    return (name="CPU", memory_gb=0.0, device_id=0)
end

function get_device_info(::CUDABackendType)
    @eval using CUDA
    dev = CUDA.device()
    mem_gb = round(CUDA.total_memory() / 1e9, digits=2)
    return (name=CUDA.name(dev), memory_gb=mem_gb, device_id=Int(dev))
end

function get_device_info(::AMDGPUBackendType)
    @eval using AMDGPU
    dev = AMDGPU.device()
    # AMDGPU memory query may vary by version
    return (name=string(dev), memory_gb=0.0, device_id=0)
end

function get_device_info(::MetalBackendType)
    @eval using Metal
    dev = Metal.device()
    return (name=string(dev), memory_gb=0.0, device_id=0)
end

# ============================================================================
# KernelAbstractions Backend
# ============================================================================

"""
    get_ka_backend(backend::GPUBackendType)

Get the KernelAbstractions backend for kernel launching.
"""
function get_ka_backend(::CPUBackendType)
    return KernelAbstractions.CPU()
end

function get_ka_backend(::CUDABackendType)
    @eval using CUDA
    return CUDABackend()
end

function get_ka_backend(::AMDGPUBackendType)
    @eval using AMDGPU
    return ROCBackend()
end

function get_ka_backend(::MetalBackendType)
    @eval using Metal
    return MetalBackend()
end

# ============================================================================
# Display
# ============================================================================

Base.show(io::IO, ::CPUBackendType) = print(io, "CPU")
Base.show(io::IO, ::CUDABackendType) = print(io, "CUDA")
Base.show(io::IO, ::AMDGPUBackendType) = print(io, "AMDGPU (ROCm)")
Base.show(io::IO, ::MetalBackendType) = print(io, "Metal")

end # module GPUBackend
