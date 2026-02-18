# Comprehensive verification: test multiple space groups
using CrystallographicFFT
using CrystallographicFFT: plan_forward, plan_backward, auto_L, fft_reconstruct!, execute_backward!
using CrystallographicFFT.SymmetryOps: SymOp, get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using FFTW

function make_symmetric(f::Array{Float64}, ops::Vector{<:SymOp}, N)
    dim = length(N)
    f_sym = zeros(Float64, N...)
    count = zeros(Int, N...)
    for ci in CartesianIndices(Tuple(N))
        x = [ci[d] - 1 for d in 1:dim]
        val = f[ci]
        for op in ops
            x_rot = Vector{Int}(undef, dim)
            for d in 1:dim
                s = 0
                for d2 in 1:dim
                    s += Int(op.R[d, d2]) * x[d2]
                end
                x_rot[d] = mod(s + round(Int, op.t[d]), N[d])
            end
            ci_new = CartesianIndex(Tuple(x_rot .+ 1))
            f_sym[ci_new] += val
            count[ci_new] += 1
        end
    end
    for ci in CartesianIndices(Tuple(N))
        if count[ci] > 0
            f_sym[ci] /= count[ci]
        end
    end
    return f_sym
end

function test_group(sg_num, N, dim)
    direct_ops = get_ops(sg_num, dim, N)
    _, shifted_ops = find_optimal_shift(direct_ops, N)
    spec_asu = calc_spectral_asu(shifted_ops, dim, N)
    
    fwd = plan_forward(Float64, spec_asu, shifted_ops)
    M = fwd.M; L = fwd.L; n_spec = fwd.n_spec
    
    # Symmetric input
    f_sym = make_symmetric(randn(Float64, N...), shifted_ops, N)
    F_full = fft(f_sym)
    
    # Extract subgrid
    f0 = zeros(Float64, M...)
    for k in 1:M[3], j in 1:M[2], i in 1:M[1]
        f0[i,j,k] = f_sym[L[1]*(i-1)+1, L[2]*(j-1)+1, L[3]*(k-1)+1]
    end
    
    # Forward
    M_vol = prod(M)
    for i in 1:M_vol; fwd.input_buffer[i] = complex(f0[i]); end
    F_cfft = fft_reconstruct!(fwd)
    
    fwd_err = 0.0
    for i in 1:n_spec
        h = get_k_vector(spec_asu, i)
        F_ref = F_full[mod(h[1],N[1])+1, mod(h[2],N[2])+1, mod(h[3],N[3])+1]
        fwd_err = max(fwd_err, abs(F_cfft[i] - F_ref))
    end
    
    # Roundtrip
    bwd = plan_backward(Float64, spec_asu, shifted_ops)
    f0_out = execute_backward!(bwd, copy(fwd.output_buffer))
    f0_rec = zeros(Float64, M...)
    for i in 1:M_vol; f0_rec[i] = real(f0_out[i]); end
    rt_err = maximum(abs, f0_rec .- f0)
    
    return M, L, n_spec, fwd_err, rt_err
end

# Test groups
test_cases = [
    (221, (16,16,16), 3, "Pm-3m (cubic)"),
    (229, (16,16,16), 3, "Im-3m (BCC cubic)"),
    (225, (16,16,16), 3, "Fm-3m (FCC cubic)"),
    (47,  (16,16,16), 3, "Pmmm (primitive orth)"),
    (2,   (16,16,16), 3, "P-1 (triclinic)"),
    (123, (16,16,16), 3, "P4/mmm (tetragonal)"),
]

println("="^100)
println("Multi-group Verification")
println("="^100)
println()

all_pass = true
for (sg, N, dim, name) in test_cases
    print("SG $sg ($name): ")
    try
        M, L, n_spec, fwd_err, rt_err = test_group(sg, N, dim)
        fwd_ok = fwd_err < 1e-8
        rt_ok = rt_err < 1e-10
        status = (fwd_ok && rt_ok) ? "✅" : "❌"
        println("$status M=$M L=$L n_spec=$n_spec fwd_err=$(round(fwd_err, sigdigits=3)) rt_err=$(round(rt_err, sigdigits=3))")
        if !fwd_ok || !rt_ok
            global all_pass = false
        end
    catch e
        println("❌ ERROR: $e")
        global all_pass = false
    end
end

println()
println(all_pass ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED")
