# bench_v3_spacegroups.jl
# Comprehensive benchmark of fractal ASU V3 across many space groups at N=64³.
# Tests correctness (V3 vs V1 max error) and speedup over full-grid FFT.
#
# Usage: julia --project=test test/bench_v3_spacegroups.jl
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_fractal_krfft, execute_fractal_krfft!,
    plan_fractal_krfft_v3, execute_fractal_krfft_v3!,
    build_recursive_tree, tree_summary, collect_leaves
using FFTW
using Statistics
using LinearAlgebra: mul!
using Printf

"""Benchmark a callable, return median time in ms."""
function bench(f; n_warmup=3, n_trials=10)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000  # ms
end

"""
    test_one_group(sg, name, N; verbose=false)

Benchmark V3 for one space group. Returns a NamedTuple with results.
"""
function test_one_group(sg, name, N; verbose=false)
    dim = length(N)
    ops = get_ops(sg, dim, N)
    G_order = length(ops)

    # Build V3 plan
    spec = calc_spectral_asu(ops, dim, N)
    plan_v3 = plan_fractal_krfft_v3(spec, ops)

    # Build V1 plan (reference)
    plan_v1 = plan_fractal_krfft(spec, ops)

    # Symmetric input for correctness
    shifted = CrystallographicFFT.KRFFT.shift_ops_half_grid(ops, collect(N), dim)
    u = rand(N...)
    s = zeros(N...)
    for op in shifted, idx in CartesianIndices(u)
        x = collect(Tuple(idx)) .- 1
        x2 = mod.(round.(Int, op.R * x + op.t), collect(N)) .+ 1
        s[idx] += u[x2...]
    end
    u_sym = s ./ length(shifted)

    # Correctness: V3 vs V1
    out_v1 = execute_fractal_krfft!(plan_v1, u_sym)
    out_v3 = execute_fractal_krfft_v3!(plan_v3, u_sym)
    max_err = maximum(abs.(out_v3 .- out_v1))

    # Use random input for timing (same work done regardless)
    u_rand = randn(Float64, N)

    # FFT baseline
    u_c = complex(u_rand)
    fft_plan = plan_fft(u_c)
    F_full = similar(u_c)
    t_fft = bench(; n_warmup=3, n_trials=10) do
        mul!(F_full, fft_plan, u_c)
    end

    # V3 timing
    t_v3 = bench(; n_warmup=3, n_trials=10) do
        execute_fractal_krfft_v3!(plan_v3, u_rand)
    end

    # Tree stats
    root = build_recursive_tree(Tuple(N), ops)
    s = tree_summary(root)
    n_spec = length(spec.points)

    return (;
        sg, name, G_order, N,
        n_spec, n_leaves=s.n_gp_leaves, n_inner=s.n_sp_nodes, depth=s.max_depth,
        max_err, t_fft, t_v3,
        speedup = t_fft / t_v3,
        theoretical_speedup = G_order,
    )
end

# ──────────────────────────────────────────────────
# Space group test suite — covering all crystal systems
# ──────────────────────────────────────────────────
const SPACE_GROUPS = [
    # Triclinic
    (2,   "P-1"),

    # Monoclinic
    (10,  "P2/m"),
    (14,  "P2₁/c"),

    # Orthorhombic
    (47,  "Pmmm"),
    (62,  "Pnma"),
    (69,  "Fmmm"),
    (71,  "Immm"),

    # Tetragonal
    (123, "P4/mmm"),
    (136, "P4₂/mnm"),
    (139, "I4/mmm"),
    (141, "I4₁/amd"),

    # Trigonal / Hexagonal
    (147, "P-3"),
    (162, "P-31m"),
    (166, "R-3m"),
    (191, "P6/mmm"),
    (194, "P6₃/mmc"),

    # Cubic (primitive)
    (195, "P23"),
    (200, "Pm-3"),
    (207, "P432"),
    (221, "Pm-3m"),
    (224, "Pn-3m"),

    # Cubic (centered)
    (225, "Fm-3m"),
    (229, "Im-3m"),
]

# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
function main()
    N = (64, 64, 64)
    vol = prod(N)

    println("=" ^ 100)
    println("Fractal ASU V3 Benchmark — N = $(N[1])³ = $(vol) points")
    println("=" ^ 100)
    println()
    @printf("%-6s %-12s %4s %8s %6s %6s %6s  %10s  %8s  %8s  %7s  %7s\n",
        "SG#", "Name", "|G|", "N_spec", "#Leaf", "#Inn", "Depth",
        "V3vsV1 err", "FFT(ms)", "V3(ms)", "Speedup", "Theory")
    println("-" ^ 100)

    results = []
    for (sg, name) in SPACE_GROUPS
        try
            r = test_one_group(sg, name, N)
            push!(results, r)

            err_str = r.max_err < 1e-10 ? "✓ $(round(r.max_err, sigdigits=2))" :
                      r.max_err < 1e-6  ? "⚠ $(round(r.max_err, sigdigits=2))" :
                                          "✗ $(round(r.max_err, sigdigits=2))"  
            @printf("%-6d %-12s %4d %8d %6d %6d %6d  %10s  %8.2f  %8.2f  %6.1f×  %6d×\n",
                r.sg, r.name, r.G_order, r.n_spec,
                r.n_leaves, r.n_inner, r.depth,
                err_str,
                r.t_fft, r.t_v3, r.speedup, r.theoretical_speedup)
        catch e
            @printf("%-6d %-12s  FAILED: %s\n", sg, name, sprint(showerror, e))
        end
    end

    # Summary
    println()
    println("=" ^ 100)
    println("SUMMARY")
    println("=" ^ 100)

    correct = filter(r -> r.max_err < 1e-10, results)
    failing = filter(r -> r.max_err >= 1e-10, results)
    faster  = filter(r -> r.speedup > 1.0, correct)

    println("  Total groups tested: $(length(results))")
    println("  Correct (err < 1e-10): $(length(correct))")
    println("  Failing (err ≥ 1e-10): $(length(failing))")
    for r in failing
        @printf("    SG %d %-12s err=%.2e\n", r.sg, r.name, r.max_err)
    end
    println("  V3 faster than FFT: $(length(faster)) / $(length(correct))")
    if !isempty(correct)
        speedups = [r.speedup for r in correct]
        @printf("  Speedup range: %.1f× – %.1f× (median %.1f×)\n",
            minimum(speedups), maximum(speedups), median(speedups))
    end
end

main()
