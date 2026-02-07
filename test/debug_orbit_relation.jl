"""
Debug: Carefully derive and verify the correct orbit FFT relation.

The DIT formula: F(h) = Σ_{x₀} ω^{h·x₀} · Y_{x₀}(h mod M)
where ω = exp(-2πi/N), M = N/L, Y_{x₀} = FFT of subgrid at shift x₀.

For symmetric f: f(R·n + t) = f(n) for all g=(R,t) in G.

Subgrid Γ_{x₀} = {x₀ + L·k : k ∈ Z_M^D}

If y₀ = R·x₀ + t mod L, then:
f(y₀ + L·k) = f(R·x₀ + t + L·k)

We need: R·(x₀ + L·k') + t = y₀ + L·k  (mod N) for some k↔k' bijection
=> R·x₀ + L·R·k' + t ≡ R·x₀ + t + L·k (mod N)  [if y₀ = R·x₀+t mod L]

Wait, y₀ = R·x₀ + t mod L means R·x₀ + t = y₀ + L·m for some integer m.
So: R·(x₀ + L·k') + t = R·x₀ + t + L·R·k' = y₀ + L·m + L·R·k' = y₀ + L·(m + R·k')

And since f is symmetric:
f(y₀ + L·(m + R·k')) = f(R⁻¹(y₀ + L·(m + R·k') - t))
= f(R⁻¹(R·x₀ + L·m + L·m + L·R·k' - ... ))

Actually, let me just be concrete:
f(y₀ + L·k) = f(R⁻¹(y₀ + L·k - t)) [by symmetry, using g⁻¹]
= f(R⁻¹·y₀ - R⁻¹·t + L·R⁻¹·k)

Since y₀ = (R·x₀ + t) mod L → y₀ = R·x₀ + t - L·m for some integer m
=> R⁻¹·y₀ = x₀ + R⁻¹·t - L·R⁻¹·m
=> R⁻¹·y₀ - R⁻¹·t = x₀ - L·R⁻¹·m

So: f(y₀ + L·k) = f(x₀ - L·R⁻¹·m + L·R⁻¹·k)
                 = f(x₀ + L·(R⁻¹·k - R⁻¹·m))   [since f is periodic with period N]

Let k' = R⁻¹·k - R⁻¹·m mod M = R⁻¹·(k - m) mod M

Y_{y₀}(q) = Σ_k f(y₀ + L·k) · exp(-2πi q·k / M)
           = Σ_k f(x₀ + L·(R⁻¹(k-m) mod M)) · exp(-2πi q·k / M)

Substitute j = R⁻¹(k-m) mod M, so k = R·j + m mod M:
= Σ_j f(x₀ + L·j) · exp(-2πi q·(R·j + m) / M)
= exp(-2πi q·m / M) · Σ_j f(x₀ + L·j) · exp(-2πi (R^T·q)·j / M)
= exp(-2πi q·m / M) · Y_{x₀}(R^T·q mod M)

KEY RESULT: Y_{y₀}(q) = exp(-2πi q·m/M) · Y_{x₀}(R^T q mod M)
where m = (R·x₀ + t - y₀) / L  (the integer quotient!)

This is the CORRECTED relation — there's an extra phase factor from m!
"""

using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using FFTW
using LinearAlgebra

N = (16, 16, 16)
L = (4, 4, 4)
n_sub = N .÷ L  # M = (4, 4, 4)
ops = get_ops(47, 3, N)

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

# Compute ALL subgrid FFTs
all_ffts = Dict{Vector{Int}, Array{ComplexF64, 3}}()
for x0_tuple in Iterators.product([0:l-1 for l in L]...)
    x0 = collect(x0_tuple)
    ranges = Tuple(x0[d]+1 : L[d] : N[d] for d in 1:3)
    all_ffts[x0] = fft(Complex.(u[ranges...]))
end

# Verify the CORRECTED relation:
# Y_{y₀}(q) = exp(-2πi q·m/M) · Y_{rep}(R^T q mod M)
# where m = (R·rep + t - y₀) / L

orbits = analyze_interleaved_orbits(N, ops; L=L)

max_err = 0.0
n_checked = 0

for orbit in orbits
    rep = orbit.representative
    Y_rep = all_ffts[rep]
    M = collect(n_sub)
    
    for (i, member) in enumerate(orbit.members)
        if member == rep
            continue
        end
        
        Y_member = all_ffts[member]
        
        # Find operation g
        found_op = nothing
        for op in orbit.ops[i]
            x_prime = mod.(op.R * rep .+ op.t, collect(L))
            if x_prime == member
                found_op = op
                break
            end
        end
        
        if isnothing(found_op)
            println("WARNING: No op found for member $member")
            continue
        end
        
        R_g = found_op.R
        t_g = found_op.t
        
        # Compute m = (R·rep + t - member) / L (element-wise integer division)
        m_vec = (R_g * rep .+ t_g .- member) .÷ collect(L)
        
        # Verify: R·rep + t = member + L·m
        @assert R_g * rep .+ t_g == member .+ collect(L) .* m_vec "m calculation wrong"
        
        # Test corrected relation at all frequencies
        local_err = 0.0
        for q_tuple in CartesianIndices(Y_rep)
            q = collect(Tuple(q_tuple)) .- 1  # 0-based
            
            # Phase correction: exp(-2πi q·m/M)
            phase = sum(q .* m_vec ./ M)
            correction = exp(-2π * im * phase)
            
            # Rotated frequency: R^T q mod M, 1-based
            rot_q = mod.(transpose(R_g) * q, M) .+ 1
            
            predicted = correction * Y_rep[rot_q...]
            actual = Y_member[q_tuple]
            
            err = abs(predicted - actual)
            local_err = max(local_err, err)
        end
        
        if local_err > max_err
            global max_err = local_err
        end
        global n_checked += 1
        
        if local_err > 1e-10
            println("FAIL: rep=$rep→member=$member, m=$m_vec, err=$local_err")
        end
    end
end

println("\nChecked $n_checked member-rep pairs")
println("Max relation error: $max_err")
println("CORRECTED relation holds: $(max_err < 1e-10)")

if max_err < 1e-10
    # Now test full orbit-reduced DIT reconstruction
    println("\n=== Full orbit-reduced DIT reconstruction ===")
    
    F_ref = fft(u)
    
    # Build lookup: shift → (rep, R_g, t_g, m_vec)
    shift_info = Dict{Vector{Int}, Tuple{Vector{Int}, Matrix{Int}, Vector, Vector{Int}}}()
    
    for orbit in orbits
        rep = orbit.representative
        for (i, member) in enumerate(orbit.members)
            if member == rep
                r_I = Matrix{Int}(I, 3, 3)
                shift_info[member] = (rep, r_I, zeros(Int, 3), zeros(Int, 3))
            else
                for op in orbit.ops[i]
                    x_prime = mod.(op.R * rep .+ op.t, collect(L))
                    if x_prime == member
                        m_vec = (op.R * rep .+ op.t .- member) .÷ collect(L)
                        shift_info[member] = (rep, op.R, op.t, m_vec)
                        break
                    end
                end
            end
        end
    end
    
    # Compute rep FFTs only
    rep_ffts = Dict{Vector{Int}, Array{ComplexF64, 3}}()
    for orbit in orbits
        rep_ffts[orbit.representative] = all_ffts[orbit.representative]
    end
    
    max_recon_err = 0.0
    for h_tuple in CartesianIndices(F_ref)
        h = collect(Tuple(h_tuple)) .- 1
        h_local = mod.(h, collect(n_sub))  # 0-based
        
        val = zero(ComplexF64)
        for x0_tuple in Iterators.product([0:l-1 for l in L]...)
            x0 = collect(x0_tuple)
            
            # Twiddle factor
            twiddle = exp(-2π * im * sum(h .* x0 ./ collect(N)))
            
            # Orbit info
            rep, R_g, t_g, m_vec = shift_info[x0]
            
            # Corrected relation: 
            # Y_{x0}(h_local) = exp(-2πi h_local·m/M) · Y_{rep}(R_g^T h_local mod M)
            phase_correction = exp(-2π * im * sum(h_local .* m_vec ./ collect(n_sub)))
            rot_h = mod.(transpose(R_g) * h_local, collect(n_sub)) .+ 1
            
            val += twiddle * phase_correction * rep_ffts[rep][rot_h...]
        end
        
        err = abs(val - F_ref[h_tuple])
        if err > max_recon_err
            global max_recon_err = err
        end
    end
    
    println("Max reconstruction error: $max_recon_err")
    println("Orbit-reduced DIT reconstruction correct: $(max_recon_err < 1e-10)")
end
