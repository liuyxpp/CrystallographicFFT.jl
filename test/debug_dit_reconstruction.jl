"""
Validate DIT reconstruction formula with orbit reduction for symmetric data.

Step 1: Verify plain DIT formula (no symmetry) → should be exact
Step 2: Verify orbit-reduced DIT formula for symmetric data → should be exact
"""

using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using FFTW
using LinearAlgebra

N = (16, 16, 16)
L = (4, 4, 4)
n_sub = N .÷ L  # (4, 4, 4)

ops = get_ops(47, 3, N)  # Pmmm, |G|=8
println("Pmmm: $(length(ops)) operations")

# Create Pmmm-symmetric data
u_raw = rand(Float64, N)
u = zeros(Float64, N)
for i in CartesianIndices(u)
    idx = collect(Tuple(i) .- 1)
    val = 0.0
    for op in ops
        new_idx = mod.(op.R * idx, collect(N))
        new_i = CartesianIndex(Tuple(new_idx .+ 1)...)
        val += u_raw[new_i]
    end
    u[i] = val / length(ops)
end

# Reference FFT
F_ref = fft(u)

# ========== Step 1: Plain DIT (all L^D subgrids) ==========
println("\n=== Step 1: Plain DIT (all $(prod(L)) subgrids) ===")

# Compute all subgrid FFTs
subgrid_ffts = Dict{Vector{Int}, Array{ComplexF64, 3}}()
for x0_tuple in Iterators.product([0:l-1 for l in L]...)
    x0 = collect(x0_tuple)
    ranges = Tuple(x0[d]+1 : L[d] : N[d] for d in 1:3)
    subgrid = Complex.(u[ranges...])
    subgrid_ffts[x0] = fft(subgrid)
end

# Reconstruct using DIT formula
max_err_dit = 0.0
for h_tuple in CartesianIndices(F_ref)
    h = collect(Tuple(h_tuple)) .- 1  # 0-based
    
    val = zero(ComplexF64)
    for (x0, Y) in subgrid_ffts
        # Twiddle factor: exp(-2πi h·x₀/N)
        phase = sum(h .* x0 ./ collect(N))
        twiddle = exp(-2π * im * phase)
        
        # Local frequency: h mod n_sub (1-based indexing)
        h_local = mod.(h, collect(n_sub)) .+ 1
        
        val += twiddle * Y[h_local...]
    end
    
    err = abs(val - F_ref[h_tuple])
    if err > max_err_dit
        global max_err_dit = err
    end
end
println("Max error: $max_err_dit")
println("DIT formula correct: $(max_err_dit < 1e-10)")

# ========== Step 2: Orbit-reduced DIT ==========
println("\n=== Step 2: Orbit-reduced DIT (symmetric data) ===")

# Analyze orbits
orbits = analyze_interleaved_orbits(N, ops; L=L)
println("Number of orbits: $(length(orbits))")
println("Total subgrids: $(prod(L))")
println("Speedup (FFT count): $(prod(L) / length(orbits))")

# Compute FFTs only for orbit representatives
rep_ffts = Dict{Vector{Int}, Array{ComplexF64, 3}}()
for orbit in orbits
    rep = orbit.representative
    ranges = Tuple(rep[d]+1 : L[d] : N[d] for d in 1:3)
    subgrid = Complex.(u[ranges...])
    rep_ffts[rep] = fft(subgrid)
end

# Build orbit lookup: for each shift x₀, find (orbit_rep, operation g mapping rep→x₀)
shift_to_orbit = Dict{Vector{Int}, Tuple{Vector{Int}, Matrix{Int}, Vector}}()
for orbit in orbits
    rep = orbit.representative
    for (i, member) in enumerate(orbit.members)
        if member == rep
            # Identity maps rep to itself
            shift_to_orbit[member] = (rep, Matrix{Int}(I, 3, 3), zeros(Int, 3))
        else
            # Find operation g such that R_g * rep + t_g ≡ member (mod L)
            for op_list in [orbit.ops[i]]
                for op in op_list
                    x_prime = mod.(op.R * rep .+ op.t, collect(L))
                    if x_prime == member
                        shift_to_orbit[member] = (rep, op.R, op.t)
                        break
                    end
                end
            end
        end
    end
end

# Reconstruct using orbit-reduced DIT formula
max_err_orbit = 0.0
for h_tuple in CartesianIndices(F_ref)
    h = collect(Tuple(h_tuple)) .- 1  # 0-based
    h_local = mod.(h, collect(n_sub))  # 0-based local frequency
    
    val = zero(ComplexF64)
    for x0_tuple in Iterators.product([0:l-1 for l in L]...)
        x0 = collect(x0_tuple)
        
        # Twiddle factor: exp(-2πi h·x₀/N)
        phase = sum(h .* x0 ./ collect(N))
        twiddle = exp(-2π * im * phase)
        
        # Look up orbit info for this shift
        rep, R_g, t_g = shift_to_orbit[x0]
        
        # F_{x₀}(h_local) = F_{rep}(R_g^T h_local mod n_sub)
        rotated_h = mod.(transpose(R_g) * h_local, collect(n_sub)) .+ 1  # 1-based
        
        Y_rep = rep_ffts[rep]
        val += twiddle * Y_rep[rotated_h...]
    end
    
    err = abs(val - F_ref[h_tuple])
    if err > max_err_orbit
        global max_err_orbit = err
    end
end
println("Max error: $max_err_orbit")
println("Orbit-reduced DIT correct: $(max_err_orbit < 1e-10)")
