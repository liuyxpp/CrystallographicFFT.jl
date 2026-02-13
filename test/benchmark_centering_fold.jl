using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, detect_centering_type
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT: plan_krfft_centered, execute_centered_krfft!
using CrystallographicFFT.KRFFT: plan_krfft, pack_stride!, fft_reconstruct!, execute_krfft!
using CrystallographicFFT.KRFFT: CenteredKRFFTPlan, fft_reconstruct_centered!, pack_stride_real!
using FFTW
using LinearAlgebra: mul!
using Random
using Printf

"""Generate symmetric field."""
function make_symmetric(ops, N)
    Random.seed!(42)
    u = randn(N...)
    u_sym = zeros(N...)
    Nv = collect(Int, N)
    for op in ops
        R = round.(Int, op.R); t = round.(Int, op.t)
        for iz in 0:N[3]-1, iy in 0:N[2]-1, ix in 0:N[1]-1
            y = mod.([sum(R[d,:].*[ix,iy,iz])+t[d] for d in 1:3], Nv)
            u_sym[y[1]+1,y[2]+1,y[3]+1] += u[ix+1,iy+1,iz+1]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

function benchmark_one(sg, name, N_size, fft_plan, fft_out; nruns=20)
    N = (N_size, N_size, N_size)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, 3, N)
    u = make_symmetric(ops_s, N)
    cent = detect_centering_type(ops_s, N)

    # Full FFT baseline: pre-planned mul! (no allocation)
    u_c = complex(u)  # convert to ComplexF64 once (not timed)
    mul!(fft_out, fft_plan, u_c)  # warmup
    F_ref = copy(fft_out)
    t_fft = minimum([@elapsed(mul!(fft_out, fft_plan, u_c)) for _ in 1:nruns])

    # Centered KRFFT
    plan_c = plan_krfft_centered(spec, ops_s)
    if !(plan_c isa CenteredKRFFTPlan)
        @printf "%-10s SG%-3d  %s  NOT APPLICABLE (fell back to plain KRFFT)\n" name sg cent
        return
    end

    # Warmup
    execute_centered_krfft!(plan_c, u)

    # Benchmark: full pipeline (pack_stride + fold + FFT + recon)
    t_full = minimum([@elapsed(execute_centered_krfft!(plan_c, u)) for _ in 1:nruns])

    # Benchmark: SCFT fast path (fold + FFT + recon, f₀ already packed)
    pack_stride_real!(plan_c.f0_buffer, u)
    t_scft = minimum([@elapsed(fft_reconstruct_centered!(plan_c)) for _ in 1:nruns])

    n_ch = plan_c.fold_plan.n_channels
    speedup_full = t_fft / t_full
    speedup_scft = t_fft / t_scft

    # Verify correctness
    execute_centered_krfft!(plan_c, u)
    spec_out = plan_c.krfft_plan.output_buffer
    max_err = 0.0
    for (i, _) in enumerate(spec.points)
        h = get_k_vector(spec, i)
        fref = F_ref[mod(h[1],N[1])+1, mod(h[2],N[2])+1, mod(h[3],N[3])+1]
        max_err = max(max_err, abs(spec_out[i] - fref))
    end

    @printf "%-10s SG%-3d  %s  %dch  N=%d  fft=%.2fms  full=%.2fms(%5.1f×)  scft=%.2fms(%5.1f×)  err=%.1e\n" name sg cent n_ch N_size t_fft*1e3 t_full*1e3 speedup_full t_scft*1e3 speedup_scft max_err
end

function main()
    println("=" ^110)
    println("Centering Fold Benchmark (FFT baseline: planned mul!, no alloc)")
    println("=" ^110)
    println()

    test_cases = [
        (70,  "Fddd"),
        (229, "Im-3m"),
        (225, "Fm-3m"),
        (227, "Fd-3m"),
        (139, "I4/mmm"),
        (63,  "Cmcm"),
        (72,  "Ibam"),
        (74,  "Imma"),
        (230, "Ia-3d"),
    ]

    for N_size in [32, 64, 128]
        println("--- N=$N_size ---")
        # Pre-plan FFT for this grid size (shared across all groups)
        u_tmp = complex(randn(N_size, N_size, N_size))
        fft_out = similar(u_tmp)
        fft_plan = plan_fft(u_tmp)
        for (sg, name) in test_cases
            benchmark_one(sg, name, N_size, fft_plan, fft_out)
        end
        println()
    end
end

main()
