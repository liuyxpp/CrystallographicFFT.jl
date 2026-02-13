# G0 ASU 性能基准测试
#
# 对比 G0 ASU (自动分派: 立方→stride-4, 非立方→通用stride-L) 与:
#   1. 全网格 FFT 基线 (FFTW C2C on N³)
#   2. plan_krfft (通用 KRFFT, stride-L 查表重构)
#
# 用法: julia --project=test test/bench_general_g0_asu.jl

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_krfft, fft_reconstruct!,
                                  plan_krfft_g0asu, execute_g0asu_krfft!,
                                  plan_krfft_g0asu_general, execute_general_g0asu_krfft!,
                                  GeneralG0ASUPlan, G0ASUPlan, auto_L
using FFTW
using Statistics
using LinearAlgebra: mul!

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

function bench_method(f; n_warmup=5, n_trials=30)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000  # ms
end

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

function benchmark_group(sg, name, N; n_warmup=5, n_trials=30)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, 3, N)
    n_spec = length(spec.points)
    n_G = length(ops_s)
    u = make_symmetric_field(ops_s, N)

    # Baseline: full FFT
    u_c = complex(u)
    fft_plan = plan_fft(u_c)
    F_full = similar(u_c)
    t_fft = bench_method(; n_warmup, n_trials) do
        mul!(F_full, fft_plan, u_c)
    end

    # G0 ASU — auto-dispatch: plan_krfft_g0asu returns G0ASUPlan for cubic, GeneralG0ASUPlan for others
    L = auto_L(ops_s)
    n_sub = prod(L)

    t_g0asu = NaN
    g0_type = "N/A"
    effective_L = L
    effective_fft_reduction = n_sub

    plan_g0 = plan_krfft_g0asu(spec, ops_s)

    if plan_g0 isa G0ASUPlan
        g0_type = "cubic-s4"
        effective_L = [4,4,4]
        effective_fft_reduction = 64

        t_g0asu = bench_method(; n_warmup, n_trials) do
            execute_g0asu_krfft!(plan_g0, spec, u)
        end
    elseif plan_g0 isa GeneralG0ASUPlan
        g0_type = "general"

        t_g0asu = bench_method(; n_warmup, n_trials) do
            execute_general_g0asu_krfft!(plan_g0, spec, u)
        end
    end

    # plan_krfft for comparison
    plan_k = plan_krfft(spec, ops_s)
    t_krfft = bench_method(; n_warmup, n_trials) do
        pack_stride_L!(plan_k, u)
        fft_reconstruct!(plan_k)
    end

    GC.gc()

    return (sg=sg, name=name, G=n_G, n_spec=n_spec,
            L=effective_L, fft_reduction=effective_fft_reduction,
            g0_type=g0_type,
            t_fft=t_fft, t_g0asu=t_g0asu, t_krfft=t_krfft,
            asu_pct=round(n_spec / prod(N) * 100, digits=1))
end

function run_benchmarks()
    test_groups = [
        # Non-cubic: use GeneralG0ASUPlan
        (2,   "P-1"),
        (10,  "P2/m"),
        (47,  "Pmmm"),
        (123, "P4/mmm"),
        # Cubic: auto-dispatch to G0ASUPlan (stride-4)
        (200, "Pm-3"),
        (221, "Pm-3m"),
    ]

    FFTW.set_num_threads(1)

    for N_size in [64, 128]
        N = (N_size, N_size, N_size)

        println("\n" * "="^130)
        println("G0 ASU Benchmark: N = $N")
        println("FFTW threads: $(FFTW.get_num_threads()), Warmup: 5, Trials: 30, Metric: median")
        println("="^130)

        header = rpad("SG", 5) * rpad("Name", 10) * rpad("|G|", 5) *
                 rpad("G0 type", 12) * rpad("L", 12) *
                 rpad("n_spec", 8) * rpad("ASU%", 7) *
                 rpad("FFT(ms)", 10) * rpad("G0(ms)", 10) *
                 rpad("KRFFT(ms)", 11) *
                 rpad("G0/FFT", 8) * rpad("KR/FFT", 8) *
                 "FFT_red"
        println(header)
        println("-"^130)

        results = []
        for (sg, name) in test_groups
            try
                r = benchmark_group(sg, name, N)
                push!(results, r)

                g0_speedup = isnan(r.t_g0asu) ? "N/A" : string(round(r.t_fft / r.t_g0asu, digits=2)) * "x"
                kr_speedup = string(round(r.t_fft / r.t_krfft, digits=2)) * "x"

                line = rpad(string(r.sg), 5) *
                       rpad(r.name, 10) *
                       rpad(string(r.G), 5) *
                       rpad(r.g0_type, 12) *
                       rpad(string(r.L), 12) *
                       rpad(string(r.n_spec), 8) *
                       rpad(string(r.asu_pct) * "%", 7) *
                       rpad(string(round(r.t_fft, digits=2)), 10) *
                       rpad(isnan(r.t_g0asu) ? "N/A" : string(round(r.t_g0asu, digits=3)), 10) *
                       rpad(string(round(r.t_krfft, digits=3)), 11) *
                       rpad(g0_speedup, 8) *
                       rpad(kr_speedup, 8) *
                       string(r.fft_reduction) * "x"
                println(line)
            catch e
                println(rpad(string(sg), 5) * rpad(name, 10) *
                        "ERROR: $(sprint(showerror, e))")
                showerror(stdout, e, catch_backtrace())
                println()
            end
        end

        println("\n--- Markdown ---")
        println("| SG | Name | |G| | G0 type | L | n_spec | ASU% | FFT (ms) | G0 ASU (ms) | KRFFT (ms) | G0/FFT | KR/FFT | FFT reduction |")
        println("|-----|------|-----|---------|---|--------|------|----------|-------------|------------|--------|--------|---------------|")
        for r in results
            g0s = isnan(r.t_g0asu) ? "N/A" : "**$(round(r.t_fft / r.t_g0asu, digits=2))x**"
            krs = "$(round(r.t_fft / r.t_krfft, digits=2))x"
            g0t = isnan(r.t_g0asu) ? "N/A" : "$(round(r.t_g0asu, digits=3))"
            println("| $(r.sg) | $(r.name) | $(r.G) | $(r.g0_type) | $(r.L) | $(r.n_spec) | $(r.asu_pct)% | $(round(r.t_fft, digits=2)) | $g0t | $(round(r.t_krfft, digits=3)) | $g0s | $krs | $(r.fft_reduction)x |")
        end
    end
end

run_benchmarks()
