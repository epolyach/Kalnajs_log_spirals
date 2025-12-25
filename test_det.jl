using Pkg
Pkg.activate(".")
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using KalnajsLogSpiral
using CUDA
using Printf

const GPU_TYPE = KalnajsLogSpiral.GPUBackend.detect_gpu_type()
println("GPU Type: ", GPU_TYPE)

config = KalnajsLogSpiral.load_config("configs/default.toml")
println("Config loaded")

model = KalnajsLogSpiral.create_toomre_zang_model(config)
println("Model created")

println("Computing precomputed data...")
precomputed = KalnajsLogSpiral.precompute_all(config, model, GPU_TYPE; verbose=true)
println("Precomputation done")

# Check orbit data for NaNs
orbit_data = precomputed.orbit_data
println("\n=== Checking orbit data for NaNs ===")
println("Omega_1 has NaN: ", any(isnan.(orbit_data.Omega_1)), " count: ", sum(isnan.(orbit_data.Omega_1)))
println("Omega_2 has NaN: ", any(isnan.(orbit_data.Omega_2)), " count: ", sum(isnan.(orbit_data.Omega_2)))
println("FE has NaN: ", any(isnan.(orbit_data.FE)), " count: ", sum(isnan.(orbit_data.FE)))
println("FL has NaN: ", any(isnan.(orbit_data.FL)), " count: ", sum(isnan.(orbit_data.FL)))
println("jacobian has NaN: ", any(isnan.(orbit_data.jacobian)), " count: ", sum(isnan.(orbit_data.jacobian)))

# Check precomputed data
println("\n=== Checking precomputed data for NaNs ===")
println("W_l_mat has NaN: ", any(isnan.(precomputed.W_l_mat)))
println("DJ_vec has NaN: ", any(isnan.(precomputed.DJ_vec)), " count: ", sum(isnan.(precomputed.DJ_vec)))
println("F0l_all has NaN: ", any(isnan.(precomputed.F0l_all)), " count: ", sum(isnan.(precomputed.F0l_all)))
println("Omega_res has NaN: ", any(isnan.(precomputed.Omega_res)), " count: ", sum(isnan.(precomputed.Omega_res)))

# Show some values
println("\n=== Sample values ===")
println("Omega_1[1:5, 1] = ", orbit_data.Omega_1[1:5, 1])
println("Omega_2[1:5, 1] = ", orbit_data.Omega_2[1:5, 1])
println("FE[1:5, 1] = ", orbit_data.FE[1:5, 1])
println("FL[1:5, 1] = ", orbit_data.FL[1:5, 1])

# Find where NaNs are
if any(isnan.(orbit_data.Omega_1))
    nan_idx = findfirst(isnan.(orbit_data.Omega_1))
    println("\nFirst NaN in Omega_1 at: ", nan_idx)
end
if any(isnan.(orbit_data.Omega_2))
    nan_idx = findfirst(isnan.(orbit_data.Omega_2))
    println("First NaN in Omega_2 at: ", nan_idx)
end
