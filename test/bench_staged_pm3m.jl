# Staged KRFFT Benchmark for Pm-3m (SG 221)
#
# Compares three approaches at N = 32, 64, 128 (full grid N³):
#   1. FFTW full-grid FFT (baseline)
#   2. G0 ASU KRFFT (plan_krfft_g0asu) → n_spec = N³/48 spectral values
#   3. Staged KRFFT (A8 + P3c + S₃ pruned butterfly + A8 combination)
#      → same n_spec = N³/48 spectral values, identical to G0 ASU output
#
# All benchmarks use single-threaded FFTW for fair comparison.
#
# Usage: julia --project=test test/bench_staged_pm3m.jl

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT: plan_krfft_g0asu, execute_g0asu_krfft!
using FFTW
using Statistics
using LinearAlgebra: mul!
using Printf

# Force single-threaded FFTW
FFTW.set_num_threads(1)

# Include staged KRFFT (standalone file, not yet in package)
include(joinpath(@__DIR__, "..", "src", "staged_krfft.jl"))

# ─── Helpers ──────────────────────────────────────────────────────────────

"""Benchmark a callable, return median time in ms."""
function bench(f; n_warmup=2, n_trials=10)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000  # ms
end

"""Generate Pm-3m symmetric data on N³ grid via real-space averaging."""
function make_symmetric(N::NTuple{3,Int}, ops)
    u_raw = randn(N...)
    u_sym = zeros(N...)
    N_arr = collect(N)
    for k in 0:N[3]-1, j in 0:N[2]-1, i in 0:N[1]-1
        val = 0.0
        for op in ops
            x2 = mod.(round.(Int, op.R * [i,j,k] .+ op.t), N_arr)
            val += u_raw[x2[1]+1, x2[2]+1, x2[3]+1]
        end
        u_sym[i+1, j+1, k+1] = val / length(ops)
    end
    return u_sym
end

# ─── Benchmark ────────────────────────────────────────────────────────────

function run_benchmark(N::Int)
    Nt = (N, N, N)
    N2 = N ÷ 2

    # Adaptive trial count
    n_trials = N <= 64 ? 10 : 5
    n_warmup = 2

    ops = get_ops(221, 3, Nt)
    _, ops_s = find_optimal_shift(ops, Nt)

    # Generate symmetric input
    u = make_symmetric(Nt, ops_s)

    # Full-grid spectral ASU (output target: n_spec = N³/|G|)
    spec = calc_spectral_asu(ops_s, 3, Nt)
    n_spec = length(spec.points)

    # ── 1. FFTW full-grid baseline ──
    u_c = complex(u)
    fft_plan = plan_fft(u_c, flags=FFTW.ESTIMATE)
    F_full = similar(u_c)
    t_fft = bench(; n_warmup, n_trials) do
        mul!(F_full, fft_plan, u_c)
    end

    # ── 2. G0 ASU KRFFT ──
    plan_g0 = plan_krfft_g0asu(spec, ops_s)
    t_g0 = bench(; n_warmup, n_trials) do
        execute_g0asu_krfft!(plan_g0, spec, u)
    end

    # ── 3. Staged KRFFT with S₃-aware pruning + A8 combination ──
    plan_st = plan_staged_pm3m(Nt)

    # Needed set: S₃ ASU of F₀₀₀ grid (N2³).
    # S₃ (C₃ + yx mirror, order 6) is the residual symmetry of F₀₀₀.
    # The S₃ ASU has the same point count as the full-grid spectral ASU.
    s3_needed = NTuple{3,Int}[]
    for k in 0:N2-1, j in 0:N2-1, i in 0:N2-1
        if s3_canonical(i, j, k, N2) == (i, j, k)
            push!(s3_needed, (i, j, k))
        end
    end

    pp = build_precomp_plan(plan_st.p3c_root, s3_needed)
    F000 = zeros(ComplexF64, N2, N2, N2)  

    # Precompute A8 combination points (twiddles + linear indices)
    a8_pts = build_a8_points(plan_st, spec, size(F000))
    result_staged = zeros(ComplexF64, n_spec)

    # Full staged pipeline timing
    t_staged = bench(; n_warmup, n_trials) do
        pack_a8!(plan_st, u)
        execute_p3c_forward!(plan_st.p3c_root, plan_st.a8_buf)
        execute_precomp_recon!(F000, pp)
        s3_fill_output!(F000)
        execute_a8_combination!(result_staged, F000, a8_pts)
    end

    # Forward-only timing (pack + P3c forward, no recon)
    t_fwd = bench(; n_warmup, n_trials) do
        pack_a8!(plan_st, u)
        execute_p3c_forward!(plan_st.p3c_root, plan_st.a8_buf)
    end

    # ── Correctness: verify staged output == G0 ASU output ──
    pack_a8!(plan_st, u)
    execute_p3c_forward!(plan_st.p3c_root, plan_st.a8_buf)
    execute_precomp_recon!(F000, pp)
    s3_fill_output!(F000)
    execute_a8_combination!(result_staged, F000, a8_pts)

    # G0 ASU reference
    result_g0 = execute_g0asu_krfft!(plan_g0, spec, u)

    # Full FFT reference
    mul!(F_full, fft_plan, u_c)

    max_err = 0.0
    for h_idx in 1:n_spec
        h = get_k_vector(spec, h_idx)
        ref = F_full[h[1]+1, h[2]+1, h[3]+1]
        max_err = max(max_err, abs(result_staged[h_idx] - ref))
    end
    err_str = @sprintf("%.1e", max_err)

    frac = round(length(s3_needed) / N2^3 * 100, digits=1)

    # ── Print ──
    sp_g0 = t_fft / t_g0
    sp_fwd = t_fft / t_fwd
    sp_st = t_fft / t_staged
    @printf("│ %4d │ %10.3f │ %10.3f  (%5.1f×) │ %10.3f  (%5.1f×) │ %10.3f  (%5.1f×) │ %6d │ %5.1f%% │ %8s │\n",
            N, t_fft, t_g0, sp_g0, t_fwd, sp_fwd, t_staged, sp_st, n_spec, frac, err_str)

    return (N=N, t_fft=t_fft, t_g0=t_g0, t_fwd=t_fwd, t_staged=t_staged, n_spec=n_spec)
end

# ─── Main ─────────────────────────────────────────────────────────────────

function main()
    println("Staged KRFFT Benchmark — Pm-3m (SG 221)")
    println("FFTW threads: $(FFTW.get_num_threads())")
    println()
    println("┌──────┬────────────┬───────────────────────┬───────────────────────┬───────────────────────┬────────┬───────┬──────────┐")
    println("│  N   │  FFTW (ms) │    G0 ASU (ms)        │   Forward (ms)        │   Staged (ms)         │ n_spec │ ASU % │   err    │")
    println("├──────┼────────────┼───────────────────────┼───────────────────────┼───────────────────────┼────────┼───────┼──────────┤")

    results = []
    for N in [32, 64, 128]
        r = run_benchmark(N)
        push!(results, r)
    end

    println("└──────┴────────────┴───────────────────────┴───────────────────────┴───────────────────────┴────────┴───────┴──────────┘")
    println()
    println("Notes:")
    println("  - N = full grid size (N³ grid), |G|=48")
    println("  - Forward = pack_a8! + P3c recursive forward (sub-FFTs only, no recon)")
    println("  - Staged = full pipeline: pack + forward + S₃-aware pruned butterfly")
    println("           + S₃ fill + precomputed A8 spectral combination")
    println("  - n_spec = N³/|G| = number of independent spectral coefficients")
    println("           (output identical to G0 ASU)")
    println("  - ASU % = S₃ ASU fraction of (N/2)³ = pruning density at butterfly level 0")
    println("  - err: max |F_staged(h) − FFT(u)(h)| at spectral ASU points")
end

main()
