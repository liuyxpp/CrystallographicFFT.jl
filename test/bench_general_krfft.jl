# General KRFFT (plan_krfft) Benchmark across Crystal Systems
#
# Tests the universally-applicable plan_krfft (GeneralCFFTPlan) across representative
# space groups from all 7 crystal systems. Unlike plan_krfft_g0asu (cubic-only),
# plan_krfft works for all 230 space groups.
#
# Measures:
#   - Full FFT baseline (FFTW C2C on N³)
#   - General KRFFT: stride-L pack + subgrid FFT + reconstruction
#   - Breakdown: subgrid FFT time vs reconstruction time
#
# Usage: julia --project=test test/bench_general_krfft.jl
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_krfft, fft_reconstruct!, fast_reconstruct!
using FFTW
using Statistics
using LinearAlgebra: mul!

"""Build a symmetrized real-space field for the given (shifted) symmetry operations."""
function make_symmetric_field(ops, N)
    u = rand(N...)
    u_sym = zeros(N...)
    R_mats = [Int.(op.R) for op in ops]
    t_vecs = [round.(Int, op.t) for op in ops]
    Nv = collect(Int, N)
    @inbounds for k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x1, x2, x3 = i-1, j-1, k-1
        for g in eachindex(ops)
            R = R_mats[g]; t = t_vecs[g]
            y1 = mod(R[1,1]*x1 + R[1,2]*x2 + R[1,3]*x3 + t[1], Nv[1]) + 1
            y2 = mod(R[2,1]*x1 + R[2,2]*x2 + R[2,3]*x3 + t[2], Nv[2]) + 1
            y3 = mod(R[3,1]*x1 + R[3,2]*x2 + R[3,3]*x3 + t[3], Nv[3]) + 1
            u_sym[i,j,k] += u[y1, y2, y3]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

"""
    pack_stride_L!(plan, u)

Generalized stride-L subgrid extraction from real array u into plan's input_buffer.
Handles anisotropic L (e.g., L=(2,2,1) for Pmm2, L=(2,1,1) for P-1).
"""
function pack_stride_L!(plan, u::AbstractArray{<:Real})
    N = plan.grid_N
    M = plan.subgrid_dims
    L = plan.L_factors[1]
    buf = plan.input_buffer

    idx = 1
    @inbounds for k in 1:M[3]
        kk = L[3] * (k - 1) + 1
        for j in 1:M[2]
            jj = L[2] * (j - 1) + 1
            for i in 1:M[1]
                ii = L[1] * (i - 1) + 1
                buf[idx] = complex(u[ii, jj, kk])
                idx += 1
            end
        end
    end
end

"""Benchmark a callable, return median time in ms."""
function bench_method(f; n_warmup=5, n_trials=30)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000  # ms
end

function benchmark_group(sg, name, crystal_system, N; n_warmup=5, n_trials=30)
    dim = 3
    ops = get_ops(sg, dim, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, dim, N)
    n_spec = length(spec.points)
    u = make_symmetric_field(ops_s, N)

    # --- Full FFT baseline ---
    u_c = complex(u)
    fft_plan = plan_fft(u_c)
    F_full = similar(u_c)
    t_fft = bench_method(; n_warmup, n_trials) do
        mul!(F_full, fft_plan, u_c)
    end

    # --- General KRFFT (plan_krfft) ---
    plan = plan_krfft(spec, ops_s)
    L = plan.L_factors[1]  # L per dimension
    M = plan.subgrid_dims
    n_ops = plan.n_ops

    # Determine which path is used
    path = if length(plan.phase_factors) == 3 && n_ops == 8 &&
              length(plan.output_buffer) == prod(M)
        "diag"
    else
        "gen"
    end

    # End-to-end benchmark: pack → FFT → reconstruct
    t_krfft = bench_method(; n_warmup, n_trials) do
        pack_stride_L!(plan, u)
        fft_reconstruct!(plan)
    end

    # Breakdown: reconstruct-only (assumes work_buffer already has FFT results)
    pack_stride_L!(plan, u)
    sub_in = reshape(plan.input_buffer, Tuple(M))
    sub_out = reshape(plan.work_buffer, Tuple(M))
    mul!(sub_out, plan.sub_plans[1], sub_in)

    t_recon = bench_method(; n_warmup, n_trials) do
        fast_reconstruct!(plan)
    end

    t_subfft = t_krfft - t_recon  # approximate subgrid FFT + pack time

    GC.gc()

    asu_pct = n_spec / prod(N) * 100

    return (sg=sg, name=name, system=crystal_system,
            G=length(ops_s), n_spec=n_spec, L=L,
            path=path, n_ops=n_ops,
            t_fft=t_fft, t_krfft=t_krfft,
            t_subfft=t_subfft, t_recon=t_recon,
            asu_pct=asu_pct)
end

function run_benchmarks()
    # Representative groups from all crystal systems (cubic N only)
    # Format: (SG number, Hermann-Mauguin symbol, Crystal system)
    test_groups = [
        # --- Triclinic ---
        (2,   "P-1",      "triclinic"),     # |G|=2,  L=(2,1,1)

        # --- Monoclinic ---
        (10,  "P2/m",     "monoclinic"),    # |G|=4,  L=(2,2,1)
        (6,   "Pm",       "monoclinic"),    # |G|=2,  L=(1,2,1)

        # --- Orthorhombic ---
        (47,  "Pmmm",     "orthorhombic"),  # |G|=8,  L=(2,2,2)
        (25,  "Pmm2",     "orthorhombic"),  # |G|=4,  L=(2,2,1)
        (16,  "P222",     "orthorhombic"),  # |G|=4,  L=(2,2,1)
        (70,  "Fddd",     "orthorhombic"),  # |G|=32, L=(2,2,2), F-centering

        # --- Tetragonal ---
        (123, "P4/mmm",   "tetragonal"),    # |G|=16, L=(2,2,2)
        (136, "P42/mnm",  "tetragonal"),    # |G|=16, L=(2,2,2)

        # --- Cubic ---
        (200, "Pm-3",     "cubic"),         # |G|=24, L=(2,2,2)
        (221, "Pm-3m",    "cubic"),         # |G|=48, L=(2,2,2)
        (223, "Pm-3n",    "cubic"),         # |G|=48, L=(2,2,2)
        (224, "Pn-3m",    "cubic"),         # |G|=48, L=(2,2,2)
        (225, "Fm-3m",    "cubic"),         # |G|=192, F-centering
        (227, "Fd-3m",    "cubic"),         # |G|=192, diamond
        (229, "Im-3m",    "cubic"),         # |G|=96, I-centering
        (230, "Ia-3d",    "cubic"),         # |G|=96
    ]

    for N_size in [64, 128]
        N = (N_size, N_size, N_size)

        println("\n" * "="^120)
        println("General KRFFT (plan_krfft) Benchmark: N = $N")
        println("FFTW threads: $(FFTW.get_num_threads()), Warmup: 5, Trials: 30, Metric: median")
        println("="^120)

        # Header
        println()
        header = rpad("SG", 5) * rpad("Name", 10) * rpad("System", 14) *
                 rpad("|G|", 5) * rpad("L", 12) * rpad("Path", 6) *
                 rpad("n_spec", 8) * rpad("ASU%", 7) *
                 rpad("FFT(ms)", 10) * rpad("KRFFT(ms)", 11) *
                 rpad("recon(ms)", 11) * rpad("Speedup", 8)
        println(header)
        println("-"^120)

        results = []
        for (sg, name, system) in test_groups
            try
                r = benchmark_group(sg, name, system, N)
                push!(results, r)

                speedup = r.t_fft / r.t_krfft
                line = rpad(string(r.sg), 5) *
                       rpad(r.name, 10) *
                       rpad(r.system, 14) *
                       rpad(string(r.G), 5) *
                       rpad(string(r.L), 12) *
                       rpad(r.path, 6) *
                       rpad(string(r.n_spec), 8) *
                       rpad(string(round(r.asu_pct, digits=1)) * "%", 7) *
                       rpad(string(round(r.t_fft, digits=2)), 10) *
                       rpad(string(round(r.t_krfft, digits=3)), 11) *
                       rpad(string(round(r.t_recon, digits=3)), 11) *
                       rpad(string(round(speedup, digits=2)) * "x", 8)
                println(line)
            catch e
                println(rpad(string(sg), 5) * rpad(name, 10) * rpad(system, 14) *
                        "ERROR: $(sprint(showerror, e))")
            end
        end

        # --- Summary by crystal system ---
        println("\n--- Summary by crystal system ---")
        for sys in ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "cubic"]
            sys_results = filter(r -> r.system == sys, results)
            isempty(sys_results) && continue
            best = argmax(r -> r.t_fft / r.t_krfft, sys_results)
            speedup = best.t_fft / best.t_krfft
            println("  $(rpad(sys, 14)): best = $(best.name) (|G|=$(best.G), " *
                    "L=$(best.L), $(best.path)) → $(round(speedup, digits=2))x")
        end

        # --- Markdown table for documentation ---
        println("\n--- Markdown table ---")
        println("| SG | Name | System | |G| | L | Path | n_spec | ASU% " *
                "| FFT (ms) | KRFFT (ms) | Recon (ms) | Speedup |")
        println("|-----|------|--------|-----|---|------|--------|------" *
                "|----------|------------|------------|---------|")
        for r in results
            speedup = r.t_fft / r.t_krfft
            println("| $(r.sg) | $(r.name) | $(r.system) | $(r.G) | $(r.L) " *
                    "| $(r.path) | $(r.n_spec) | $(round(r.asu_pct, digits=1))% " *
                    "| $(round(r.t_fft, digits=2)) | $(round(r.t_krfft, digits=3)) " *
                    "| $(round(r.t_recon, digits=3)) | **$(round(speedup, digits=2))x** |")
        end
    end
end

# Set single-threaded FFTW for reproducible benchmarks
FFTW.set_num_threads(1)

run_benchmarks()
