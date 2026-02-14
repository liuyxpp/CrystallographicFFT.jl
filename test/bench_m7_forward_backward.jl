"""
Benchmark: M7 forward vs backward (centered KRFFT)

Compares:
  - M7 Forward (SCFT fast path): fft_reconstruct_centered!
  - M7 Backward (SCFT fast path): ifft_unrecon_centered!
  - Full FFT baseline: plan_fft + mul!

Space groups aligned with docs/design/krfft_performance_comparison.md.
"""

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, detect_centering_type
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT
using CrystallographicFFT.KRFFT: plan_krfft_centered, CenteredKRFFTPlan
using CrystallographicFFT.KRFFT: fft_reconstruct_centered!, pack_stride_real!
using CrystallographicFFT.KRFFT: plan_centered_ikrfft, CenteredKRFFTBackwardPlan
using CrystallographicFFT.KRFFT: ifft_unrecon_centered!
using FFTW
using LinearAlgebra: mul!
using Random
using Printf
using Statistics

"""Generate symmetric field via scatter (sequential reads, random writes → cache friendly)."""
function make_symmetric(ops, N)
    Random.seed!(42)
    u = randn(N...)
    u_sym = zeros(N...)
    N1, N2, N3 = N
    R_mats = [Int.(op.R) for op in ops]
    t_vecs = [round.(Int, op.t) for op in ops]
    @inbounds for k in 1:N3, j in 1:N2, i in 1:N1
        v = u[i, j, k]
        x1, x2, x3 = i - 1, j - 1, k - 1
        for g in eachindex(ops)
            R = R_mats[g]; t = t_vecs[g]
            y1 = mod(R[1,1]*x1 + R[1,2]*x2 + R[1,3]*x3 + t[1], N1) + 1
            y2 = mod(R[2,1]*x1 + R[2,2]*x2 + R[2,3]*x3 + t[2], N2) + 1
            y3 = mod(R[3,1]*x1 + R[3,2]*x2 + R[3,3]*x3 + t[3], N3) + 1
            u_sym[y1, y2, y3] += v
        end
    end
    u_sym ./= length(ops)
    return u_sym
end


function benchmark_one(sg, name, N_size, fft_plan, fft_out; n_warmup=2, n_trials=10)
    N = (N_size, N_size, N_size)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, 3, N)
    u = make_symmetric(ops_s, N)
    cent = detect_centering_type(ops_s, N)

    # ── Full FFT baseline ──
    u_c = complex(u)
    mul!(fft_out, fft_plan, u_c)
    t_fft = minimum([@elapsed(mul!(fft_out, fft_plan, u_c)) for _ in 1:n_trials])

    # ── M7 Forward plan ──
    fwd = plan_krfft_centered(spec, ops_s)
    if !(fwd isa CenteredKRFFTPlan)
        @printf "%-10s SG%-3d  %s  N=%d  NOT APPLICABLE (fell back to plain KRFFT)\n" name sg cent N_size
        return nothing
    end

    # ── M7 Backward plan ──
    bwd = plan_centered_ikrfft(spec, ops_s, fwd)
    n_ch = fwd.fold_plan.n_channels
    M = N .÷ 2

    # Warmup
    pack_stride_real!(fwd.f0_buffer, u)
    for _ in 1:n_warmup
        fft_reconstruct_centered!(fwd)
    end
    F_spec = copy(fwd.krfft_plan.output_buffer)
    for _ in 1:n_warmup
        ifft_unrecon_centered!(bwd, F_spec)
    end

    # ── Benchmark M7 Forward (SCFT path) ──
    pack_stride_real!(fwd.f0_buffer, u)
    t_fwd = minimum([@elapsed(fft_reconstruct_centered!(fwd)) for _ in 1:n_trials])

    # ── Benchmark M7 Backward (SCFT path) ──
    t_bwd = minimum([@elapsed(ifft_unrecon_centered!(bwd, F_spec)) for _ in 1:n_trials])

    # ── Verify roundtrip ──
    pack_stride_real!(fwd.f0_buffer, u)
    f0_orig = copy(fwd.f0_buffer)
    F_check = copy(fft_reconstruct_centered!(fwd))
    f0_out = zeros(M...)
    KRFFT.execute_centered_ikrfft!(bwd, F_check, f0_out)
    rt_err = maximum(abs.(f0_out .- f0_orig))

    speedup_fwd = t_fft / t_fwd
    speedup_bwd = t_fft / t_bwd
    ratio_bwd_fwd = t_bwd / t_fwd

    @printf("%-10s SG%-3d  %s  %dch  N=%d  fft=%.2fms  fwd=%.2fms(%5.1f×)  bwd=%.2fms(%5.1f×)  bwd/fwd=%.2f  rt_err=%.1e\n",
            name, sg, cent, n_ch, N_size,
            t_fft*1e3, t_fwd*1e3, speedup_fwd,
            t_bwd*1e3, speedup_bwd, ratio_bwd_fwd, rt_err)

    return (name=name, sg=sg, cent=string(cent), n_ch=n_ch, N=N_size,
            t_fft=t_fft, t_fwd=t_fwd, t_bwd=t_bwd,
            speedup_fwd=speedup_fwd, speedup_bwd=speedup_bwd,
            ratio=ratio_bwd_fwd, rt_err=rt_err)
end

function main()
    FFTW.set_num_threads(1)

    println("=" ^120)
    println("M7 Forward vs Backward Benchmark (SCFT fast path, single-threaded FFTW)")
    println("=" ^120)
    println()

    # Space groups from krfft_performance_comparison.md (centered only)
    test_cases = [
        # Cubic centered
        (225, "Fm-3m"),
        (227, "Fd-3m"),
        (229, "Im-3m"),
        (230, "Ia-3d"),
        # Non-cubic centered
        (70,  "Fddd"),
        (139, "I4/mmm"),
        (63,  "Cmcm"),
        (72,  "Ibam"),
        (74,  "Imma"),
    ]

    all_results = []

    for (N_size, n_trials) in [(64, 10), (128, 5)]
        println("--- N=$N_size ($n_trials trials, warmup=2) ---")
        @printf("%-10s %-6s  %4s  %3s  N=%-3d  %-9s  %-16s  %-16s  %-8s  %-8s\n",
                "Group", "SG", "Cent", "Ch", N_size, "FFT(ms)", "Fwd(ms)(spdup)", "Bwd(ms)(spdup)", "Bwd/Fwd", "RT_err")
        println("-" ^120)

        u_tmp = complex(randn(N_size, N_size, N_size))
        fft_out = similar(u_tmp)
        fft_plan = plan_fft(u_tmp)

        for (sg, name) in test_cases

            r = benchmark_one(sg, name, N_size, fft_plan, fft_out;
                              n_warmup=2, n_trials=n_trials)
            r !== nothing && push!(all_results, r)
        end
        println()
    end

    # Summary table
    println()
    println("=" ^100)
    println("Summary: Backward/Forward Ratio")
    println("=" ^100)
    @printf("%-10s %-5s %-6s %8s %8s %8s %8s\n",
            "Group", "SG", "Cent", "N=64", "N=128", "N=64fwd", "N=64bwd")
    println("-" ^100)
    for (sg, name) in test_cases
        rs64 = filter(r -> r.sg == sg && r.N == 64, all_results)
        rs128 = filter(r -> r.sg == sg && r.N == 128, all_results)
        ratio64 = isempty(rs64) ? "—" : @sprintf("%.2f×", rs64[1].ratio)
        ratio128 = isempty(rs128) ? "—" : @sprintf("%.2f×", rs128[1].ratio)
        fwd64 = isempty(rs64) ? "—" : @sprintf("%.2fms", rs64[1].t_fwd*1e3)
        bwd64 = isempty(rs64) ? "—" : @sprintf("%.2fms", rs64[1].t_bwd*1e3)
        @printf("%-10s %-5d %-6s %8s %8s %8s %8s\n",
                name, sg, isempty(rs64) ? "?" : rs64[1].cent,
                ratio64, ratio128, fwd64, bwd64)
    end

    println("=" ^100)
end

main()
