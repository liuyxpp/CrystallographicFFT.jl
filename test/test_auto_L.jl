using CrystallographicFFT
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.ASU
using CrystallographicFFT.SpectralIndexing
using CrystallographicFFT.KRFFT
using FFTW

function test_sg(sg, N)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    L = auto_L(ops_s)
    spec = calc_spectral_asu(ops_s, 3, N)
    plan = plan_krfft(spec, ops_s)

    # Proper orbit averaging for symmetrization
    u = rand(N...)
    u_sym = zeros(N...)
    for op in ops_s
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(op.R * x .+ op.t, collect(N)) .+ 1
            u_sym[idx] += u[x2...]
        end
    end
    u_sym ./= length(ops_s)

    F_full = fft(complex(u_sym))
    f0 = u_sym[1:L[1]:end, 1:L[2]:end, 1:L[3]:end]
    plan.input_buffer .= vec(complex(f0))
    result = fft_reconstruct!(plan)

    max_err = 0.0
    for (h_idx, _) in enumerate(spec.points)
        h = get_k_vector(spec, h_idx)
        h_mod = mod.(h, collect(N)) .+ 1
        ref = F_full[h_mod...]
        max_err = max(max_err, abs(result[h_idx] - ref))
    end

    ok = max_err < 1e-8
    diag = length(plan.phase_factors) == 3 ? "diag" : "gen "
    println("SG $(lpad(sg,3)): |G|=$(lpad(length(ops_s),3)) L=$L ($(prod(L))x) path=$diag err=$(round(max_err,sigdigits=3)) $(ok ? "✓" : "✗")")
    return ok
end

function main()
    println("Comprehensive reconstruction test (auto_L + plan_krfft)")
    println("-"^78)
    n_pass = 0
    n_total = 0
    for sg in [47, 25, 16, 10, 2, 6, 123, 221, 229, 225, 136, 70, 223, 224, 227, 230]
        n_total += 1
        try
            if test_sg(sg, (16,16,16))
                n_pass += 1
            end
        catch e
            println("SG $(lpad(sg,3)): ERROR: $e")
        end
        GC.gc()
    end
    println("-"^78)
    println("$n_pass / $n_total PASSED")
end

main()
