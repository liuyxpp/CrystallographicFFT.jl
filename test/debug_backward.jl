## Debug P4₂/mnm forward — per-spectral-point comparison

using FFTW, Random
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT: plan_krfft_g0asu, execute_g0asu_krfft!,
                                  plan_krfft_selective, execute_selective_krfft!

function make_symmetric(ops, N)
    Random.seed!(42)
    u = rand(N...)
    s = zeros(N...)
    for op in ops
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(round.(Int, op.R) * x .+ round.(Int, op.t), collect(N)) .+ 1
            s[idx] += u[x2...]
        end
    end
    s ./= length(ops)
end

function debug_forward(sg, name, N)
    println("\n=== $name (SG $sg) N=$N ===")
    dim = 3
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, 3, N)
    u = make_symmetric(ops_s, N)

    plan = plan_krfft_g0asu(spec, ops_s)
    F_asu = execute_g0asu_krfft!(plan, spec, u)
    F_ref = fft(complex(u))

    n_spec = length(spec.points)
    println("n_spec=$n_spec, n_reps=$(plan.n_reps)")

    # Forward comparison
    n_wrong = 0
    max_err = 0.0
    for i in 1:n_spec
        hv = get_k_vector(spec, i)
        ci = CartesianIndex(Tuple(mod.(hv, collect(N)) .+ 1))
        err = abs(F_asu[i] - F_ref[ci])
        max_err = max(max_err, err)
        if err > 1e-8
            n_wrong += 1
            if n_wrong <= 5
                println("  h=$hv: F_asu=$(round(F_asu[i],digits=4)), F_ref=$(round(F_ref[ci],digits=4)), err=$(round(err,digits=4))")
            end
        end
    end
    println("Forward: max_err=$max_err, n_wrong=$n_wrong/$n_spec")

    # Test selective approach for comparison
    plan_sel = plan_krfft_selective(spec, ops_s)
    F_sel = execute_selective_krfft!(plan_sel, spec, u)

    sel_err = maximum(1:n_spec) do i
        hv = get_k_vector(spec, i)
        ci = CartesianIndex(Tuple(mod.(hv, collect(N)) .+ 1))
        abs(F_sel[i] - F_ref[ci])
    end
    println("Selective: max_err=$sel_err")

    # G0 ASU vs Selective
    asu_vs_sel = maximum(abs.(F_asu .- F_sel))
    println("G0 ASU vs Selective: max_err=$asu_vs_sel")
end

debug_forward(136, "P4₂/mnm", (16,16,16))
debug_forward(70, "Fddd", (16,16,16))
