using Pkg
Pkg.activate(".")
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using KalnajsLogSpiral
const GPU_TYPE = KalnajsLogSpiral.GPUBackend.NONE

config = KalnajsLogSpiral.load_config("configs/default.toml")
model = KalnajsLogSpiral.create_toomre_zang_model(config)

# Compute grids only
grids = KalnajsLogSpiral.OrbitIntegration.compute_grids(config, model)

println("=== Grid values at (1,1) ===")
println("Rc[1,1] = ", grids.Rc[1,1])
println("e[1,1] = ", grids.e[1,1])
println("R1[1,1] = ", grids.R1[1,1])
println("R2[1,1] = ", grids.R2[1,1])
println("E[1,1] = ", grids.E[1,1])
println("L_m[1,1] = ", grids.L_m[1,1])
println("L2_m[1,1] = ", grids.L2_m[1,1])
println("SGNL[1,1] = ", grids.SGNL[1,1])
println("v[1,1] = ", grids.v[1,1])

println("\n=== Model values at Rc[1,1] ===")
r = grids.Rc[1,1]
println("V(r) = ", KalnajsLogSpiral.Models.potential(model, r))
println("dV(r) = ", KalnajsLogSpiral.Models.potential_derivative(model, r))
println("Omega(r) = ", KalnajsLogSpiral.Models.rotation_frequency(model, r))
println("kappa(r) = ", KalnajsLogSpiral.Models.epicyclic_frequency(model, r))
