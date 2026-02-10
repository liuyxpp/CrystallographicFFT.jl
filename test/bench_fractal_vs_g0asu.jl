# Fractal ASU vs G0 ASU Benchmark
# Compares forward transform performance for non-centering space groups.
#
# Key API difference:
#   - G0 ASU: needs `find_optimal_shift` ops, data symmetric under those ops
#   - Fractal ASU: needs RAW ops (shifts internally via shift_ops_half_grid),
#                  data symmetric under b=1/2 shifted ops
#
# Usage: julia --project=test test/bench_fractal_vs_g0asu.jl
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_krfft_g0asu, execute_g0asu_krfft!,
    plan_fractal_krfft, execute_fractal_krfft!,
    shift_ops_half_grid, tree_summary, collect_leaves
using FFTW
using Statistics
using LinearAlgebra: mul!

# ── Helpers ──

"""Symmetrize under raw ops (for G0 ASU with find_optimal_shift)."""
function make_symmetric_raw(ops, N)
    u = rand(N...)
    s = zeros(N...)
    R_mats = [round.(Int, op.R) for op in ops]
    t_vecs = [round.(Int, op.t) for op in ops]
    Nv = collect(Int, N)
    @inbounds for k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x = (i-1, j-1, k-1)
        for g in eachindex(ops)
            R = R_mats[g]; t = t_vecs[g]
            y1 = mod(R[1,1]*x[1] + R[1,2]*x[2] + R[1,3]*x[3] + t[1], Nv[1]) + 1
            y2 = mod(R[2,1]*x[1] + R[2,2]*x[2] + R[2,3]*x[3] + t[2], Nv[2]) + 1
            y3 = mod(R[3,1]*x[1] + R[3,2]*x[2] + R[3,3]*x[3] + t[3], Nv[3]) + 1
            s[i,j,k] += u[y1, y2, y3]
        end
    end
    s ./= length(ops)
end

"""Symmetrize under b=1/2 shifted ops (for fractal ASU)."""
function make_symmetric_shifted(ops, N)
    D = length(N)
    shifted = shift_ops_half_grid(ops, collect(N), D)
    make_symmetric_raw(shifted, N)
end

"""Benchmark a callable, return median time in ms."""
function bench(f; n_warmup=5, n_trials=30)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    median(times) * 1000
end

# ── Benchmark functions ──

"""Benchmark for cubic groups (both G0 ASU and fractal ASU available)."""
function benchmark_cubic(sg, name, N; n_warmup=5, n_trials=30)
    dim = 3
    ops_raw = get_ops(sg, dim, N)
    _, ops_s = find_optimal_shift(ops_raw, N)
    spec_g0 = calc_spectral_asu(ops_s, dim, N)
    spec_frac = calc_spectral_asu(ops_raw, dim, N)

    # Symmetric data for each method
    u_g0 = make_symmetric_raw(ops_s, N)
    u_frac = make_symmetric_shifted(ops_raw, N)

    # --- Full FFT baseline ---
    u_c = complex(u_g0)
    fft_plan = plan_fft(u_c)
    F_full = similar(u_c)
    t_fft = bench(; n_warmup, n_trials) do
        mul!(F_full, fft_plan, u_c)
    end

    # --- G0 ASU ---
    p_g0 = plan_krfft_g0asu(spec_g0, ops_s)
    out_g0 = execute_g0asu_krfft!(p_g0, spec_g0, u_g0)
    t_g0 = bench(; n_warmup, n_trials) do
        execute_g0asu_krfft!(p_g0, spec_g0, u_g0)
    end

    # --- Fractal ASU ---
    p_frac = plan_fractal_krfft(spec_frac, ops_raw)
    out_frac = execute_fractal_krfft!(p_frac, u_frac)
    ts = tree_summary(p_frac.root)
    fft_pts = sum(prod(l.subgrid_N) for l in collect_leaves(p_frac.root))
    t_frac = bench(; n_warmup, n_trials) do
        execute_fractal_krfft!(p_frac, u_frac)
    end

    # Correctness
    F_ref_g0 = fft(u_g0)
    err_g0 = maximum(abs(out_g0[i] - F_ref_g0[(spec_g0.points[i].idx .+ 1)...])
                     for i in 1:length(spec_g0.points))
    F_ref_frac = fft(u_frac)
    err_frac = maximum(abs(out_frac[i] - F_ref_frac[(spec_frac.points[i].idx .+ 1)...])
                       for i in 1:length(spec_frac.points))

    GC.gc()

    return (sg=sg, name=name, G=length(ops_raw), N=N[1],
            n_spec=length(spec_frac.points),
            t_fft=t_fft, t_g0=t_g0, t_frac=t_frac,
            fft_pts=fft_pts, n_leaves=ts.n_gp_leaves, depth=ts.max_depth,
            err_g0=err_g0, err_frac=err_frac, has_g0=true)
end

"""Benchmark for non-cubic groups (fractal ASU only, no G0 ASU)."""
function benchmark_noncubic(sg, name, N; n_warmup=5, n_trials=30)
    dim = 3
    ops_raw = get_ops(sg, dim, N)
    spec = calc_spectral_asu(ops_raw, dim, N)

    u_frac = make_symmetric_shifted(ops_raw, N)

    # --- Full FFT baseline ---
    u_c = complex(u_frac)
    fft_plan = plan_fft(u_c)
    F_full = similar(u_c)
    t_fft = bench(; n_warmup, n_trials) do
        mul!(F_full, fft_plan, u_c)
    end

    # --- Fractal ASU ---
    p_frac = plan_fractal_krfft(spec, ops_raw)
    out_frac = execute_fractal_krfft!(p_frac, u_frac)
    ts = tree_summary(p_frac.root)
    fft_pts = sum(prod(l.subgrid_N) for l in collect_leaves(p_frac.root))
    t_frac = bench(; n_warmup, n_trials) do
        execute_fractal_krfft!(p_frac, u_frac)
    end

    # Correctness
    F_ref = fft(u_frac)
    err_frac = maximum(abs(out_frac[i] - F_ref[(spec.points[i].idx .+ 1)...])
                       for i in 1:length(spec.points))

    GC.gc()

    return (sg=sg, name=name, G=length(ops_raw), N=N[1],
            n_spec=length(spec.points),
            t_fft=t_fft, t_g0=NaN, t_frac=t_frac,
            fft_pts=fft_pts, n_leaves=ts.n_gp_leaves, depth=ts.max_depth,
            err_g0=NaN, err_frac=err_frac, has_g0=false)
end

# ── Main ──

function run_benchmarks()
    # Groups sorted by |G| descending
    # Cubic groups support both G0 ASU and Fractal ASU
    # Non-cubic groups only support Fractal ASU
    test_groups = [
        (221, "Pm-3m",   true),    # |G|=48, cubic
        (200, "Pm-3",    true),    # |G|=24, cubic
        (136, "P42/mnm", false),   # |G|=16, tetragonal
        (47,  "Pmmm",    false),   # |G|=8, orthorhombic
        (2,   "P-1",     false),   # |G|=2, triclinic
    ]

    for N_size in [64, 128]
        N = (N_size, N_size, N_size)

        println("\n" * "="^110)
        println("Grid: N=$(N_size)^3 = $(prod(N)),  Warmup=5, Trials=30, Metric=median")
        println("="^110)

        # Header
        hdr = rpad("SG", 5) * rpad("Name", 10) * rpad("|G|", 5) *
              rpad("n_spec", 8) *
              rpad("FFT(ms)", 10) *
              rpad("G0(ms)", 10) * rpad("G0_x", 8) *
              rpad("Frac(ms)", 10) * rpad("Frac_x", 8) *
              rpad("leaves", 7) * rpad("fft_pts", 10) * rpad("理论约化", 10) *
              rpad("Fr_err", 10)
        println(hdr)
        println("-"^110)

        results = []
        for (sg, name, is_cubic) in test_groups
            try
                r = is_cubic ? benchmark_cubic(sg, name, N) :
                               benchmark_noncubic(sg, name, N)
                push!(results, r)

                reduction = prod(N) / r.fft_pts
                g0_str = r.has_g0 ? string(round(r.t_g0, digits=3)) : "N/A"
                g0x_str = r.has_g0 ? string(round(r.t_fft/r.t_g0, digits=1)) * "x" : "N/A"

                line = rpad(string(r.sg), 5) *
                       rpad(r.name, 10) *
                       rpad(string(r.G), 5) *
                       rpad(string(r.n_spec), 8) *
                       rpad(string(round(r.t_fft, digits=2)), 10) *
                       rpad(g0_str, 10) * rpad(g0x_str, 8) *
                       rpad(string(round(r.t_frac, digits=3)), 10) *
                       rpad(string(round(r.t_fft/r.t_frac, digits=1)) * "x", 8) *
                       rpad(string(r.n_leaves), 7) *
                       rpad(string(r.fft_pts), 10) *
                       rpad(string(round(reduction, digits=1)) * "x", 10) *
                       rpad(string(round(r.err_frac, sigdigits=2)), 10)
                println(line)
            catch e
                println(rpad(string(sg), 5) * rpad(name, 10) *
                        "ERROR: $(sprint(showerror, e))")
            end
        end

        # Summary
        println("\n--- Summary ---")
        for r in results
            reduction = prod(N) / r.fft_pts
            sp_frac = round(r.t_fft / r.t_frac, digits=1)
            if r.has_g0
                sp_g0 = round(r.t_fft / r.t_g0, digits=1)
                ratio = round(r.t_g0 / r.t_frac, digits=2)
                println("  $(rpad(r.name, 10)): |G|=$(rpad(r.G,3)) " *
                        "G0=$(sp_g0)x  Fractal=$(sp_frac)x  " *
                        "Fractal vs G0: $(ratio)x  理论=$(round(reduction, digits=1))x")
            else
                println("  $(rpad(r.name, 10)): |G|=$(rpad(r.G,3)) " *
                        "Fractal=$(sp_frac)x  理论=$(round(reduction, digits=1))x  " *
                        "(G0 ASU不支持)")
            end
        end
    end
end

run_benchmarks()
