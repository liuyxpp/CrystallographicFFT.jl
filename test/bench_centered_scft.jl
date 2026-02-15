"""
Benchmark: CenteredSCFTPlan (fwd+bwd) vs M7+Q vs M2+Q vs Full FFT

Compares SCFT diffusion step performance across centered space groups.
"""

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_centered_scft, execute_centered_scft!,
    CenteredSCFTPlan
using CrystallographicFFT.QFusedKRFFT: plan_m2_q, execute_m2_q!,
    plan_m7_scft, execute_m7_scft!
using FFTW
using LinearAlgebra
using Printf
using Random
using Statistics

"""Generate symmetric field respecting shifted ops."""
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

function extract_subgrid(u_sym, N)
    M = N .÷ 2
    return Float64[u_sym[1+(i-1)*2, 1+(j-1)*2, 1+(k-1)*2]
                   for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

function benchmark_one(sg, N; n_warmup=5, n_trials=30)
    Δs = 0.05
    lattice = Matrix{Float64}(I, 3, 3)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    u_sym = make_symmetric(ops_s, N)
    f0_template = extract_subgrid(u_sym, N)

    # --- Full FFT baseline ---
    F_full = zeros(ComplexF64, N...)
    K_full = zeros(ComplexF64, N...)
    recip_B = 2π * inv(lattice)'
    for iz in 0:N[3]-1, iy in 0:N[2]-1, ix in 0:N[1]-1
        hc = [ix >= N[1]÷2 ? ix-N[1] : ix,
              iy >= N[2]÷2 ? iy-N[2] : iy,
              iz >= N[3]÷2 ? iz-N[3] : iz]
        kv = recip_B * hc
        K_full[ix+1,iy+1,iz+1] = exp(-dot(kv,kv) * Δs)
    end
    fft_plan = plan_fft!(F_full)
    ifft_plan = plan_ifft!(F_full)

    # Warmup full FFT
    for _ in 1:n_warmup
        copyto!(F_full, complex.(u_sym))
        fft_plan * F_full
        F_full .*= K_full
        ifft_plan * F_full
    end
    t_full = Float64[]
    for _ in 1:n_trials
        copyto!(F_full, complex.(u_sym))
        t0 = time_ns()
        fft_plan * F_full
        F_full .*= K_full
        ifft_plan * F_full
        push!(t_full, (time_ns() - t0) / 1e6)
    end

    # --- M2+Q ---
    m2q = plan_m2_q(N, sg, 3, Δs, lattice)
    f0_m2 = copy(f0_template)
    for _ in 1:n_warmup
        copyto!(f0_m2, f0_template)
        execute_m2_q!(m2q, f0_m2)
    end
    t_m2q = Float64[]
    for _ in 1:n_trials
        copyto!(f0_m2, f0_template)
        t0 = time_ns()
        execute_m2_q!(m2q, f0_m2)
        push!(t_m2q, (time_ns() - t0) / 1e6)
    end

    # --- M7+Q ---
    m7 = try
        plan_m7_scft(N, sg, 3, Δs, lattice)
    catch
        nothing
    end
    t_m7q = Float64[]
    if m7 !== nothing
        f0_m7 = copy(f0_template)
        for _ in 1:n_warmup
            copyto!(f0_m7, f0_template)
            execute_m7_scft!(m7, f0_m7)
        end
        for _ in 1:n_trials
            copyto!(f0_m7, f0_template)
            t0 = time_ns()
            execute_m7_scft!(m7, f0_m7)
            push!(t_m7q, (time_ns() - t0) / 1e6)
        end
    end

    # --- fwd+bwd ---
    spec_asu = calc_spectral_asu(ops_s, 3, N)
    scft = plan_centered_scft(spec_asu, ops_s, N, Δs, lattice)
    f0_fb = copy(f0_template)
    for _ in 1:n_warmup
        copyto!(f0_fb, f0_template)
        execute_centered_scft!(scft, f0_fb)
    end
    t_fb = Float64[]
    for _ in 1:n_trials
        copyto!(f0_fb, f0_template)
        t0 = time_ns()
        execute_centered_scft!(scft, f0_fb)
        push!(t_fb, (time_ns() - t0) / 1e6)
    end

    return (full=median(t_full), m2q=median(t_m2q),
            m7q=isempty(t_m7q) ? NaN : median(t_m7q),
            fwdbwd=median(t_fb),
            n_spec=length(spec_asu.points))
end

function run_benchmarks()
    groups = [
        (225, "Fm-3m", 192),
        (227, "Fd-3m", 192),
        (229, "Im-3m", 96),
        (230, "Ia-3d", 96),
        (70,  "Fddd",  32),
        (139, "I4/mmm", 32),
        (63,  "Cmcm",  16),
        (72,  "Ibam",  16),
        (74,  "Imma",  16),
    ]

    for N_side in [64, 128]
        N = (N_side, N_side, N_side)
        println("\n" * "="^80)
        @printf("N = %d³\n", N_side)
        println("="^80)
        @printf("%-8s %5s %6s  %8s %8s %8s %8s  %6s %6s\n",
                "Group", "|G|", "n_spec",
                "Full", "M2+Q", "M7+Q", "fwd+bwd",
                "fb/M7Q", "fb/Full")
        println("-"^80)

        for (sg, name, g_order) in groups
            try
                r = benchmark_one(sg, N)
                fb_vs_m7q = isnan(r.m7q) ? NaN : r.fwdbwd / r.m7q
                fb_vs_full = r.fwdbwd / r.full
                @printf("%-8s %5d %6d  %7.2f %7.2f %7.2f %7.2f  %5.2f× %5.2f×\n",
                        name, g_order, r.n_spec,
                        r.full, r.m2q, r.m7q, r.fwdbwd,
                        fb_vs_m7q, fb_vs_full)
            catch e
                @printf("%-8s %5d    —   FAILED: %s\n", name, g_order, string(e))
            end
        end
    end
end

run_benchmarks()
