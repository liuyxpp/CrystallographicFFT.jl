"""
Benchmark: M2 fwd+bwd vs M2+Q SCFT

Compares:
  - M2 fwd+bwd: plan_m2_scft + execute_m2_scft!  (spectral K multiply)
  - M2+Q:       plan_m2_q + execute_m2_q!          (Q-matrix multiply)
  - Full FFT:   plan_fft + mul! baseline

Covers both P-centering (M2 only) and centered groups.
"""

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: auto_L
using CrystallographicFFT.QFusedKRFFT: plan_m2_scft, execute_m2_scft!,
    plan_m2_q, execute_m2_q!
using FFTW
using LinearAlgebra
using Random
using Printf
using Statistics

"""Generate symmetric field (allocation-free inner loop)."""
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

"""Extract stride-L subgrid."""
function extract_subgrid(u, N, L)
    M = N .÷ Tuple(L)
    Float64[u[1+(i-1)*L[1], 1+(j-1)*L[2], 1+(k-1)*L[3]]
            for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

"""Full-grid FFT diffusion reference (for accuracy check)."""
function fullgrid_diffuse(u_sym, N, Δs, lattice)
    F = fft(u_sym)
    rB = 2π * inv(Matrix(lattice))'
    N1, N2, N3 = N
    @inbounds for iz in 0:N3-1, iy in 0:N2-1, ix in 0:N1-1
        hc = [ix>=N1÷2 ? ix-N1 : ix, iy>=N2÷2 ? iy-N2 : iy, iz>=N3÷2 ? iz-N3 : iz]
        kv = rB * hc
        F[ix+1,iy+1,iz+1] *= exp(-dot(kv,kv)*Δs)
    end
    real.(ifft(F))
end

function benchmark_one(sg, name, N_size, fft_plan, fft_out;
                       n_warmup=3, n_trials=10)
    N = (N_size, N_size, N_size)
    Δs = 0.05
    lattice = Matrix{Float64}(I, 3, 3)

    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    L = auto_L(ops_s)
    spec = calc_spectral_asu(ops_s, 3, N)
    n_spec = length(spec.points)
    u_sym = make_symmetric(ops_s, N)

    M = N .÷ Tuple(L)
    L_str = join(L, "×")

    # ── Full FFT baseline ──
    u_c = complex(u_sym)
    mul!(fft_out, fft_plan, u_c)
    t_fft = minimum([@elapsed(mul!(fft_out, fft_plan, u_c)) for _ in 1:n_trials])

    # ── M2 fwd+bwd ──
    m2fb = plan_m2_scft(N, sg, 3, Δs, lattice)
    f0_fb = extract_subgrid(u_sym, N, L)
    for _ in 1:n_warmup
        f0_tmp = copy(f0_fb)
        execute_m2_scft!(m2fb, f0_tmp)
    end
    times_fb = Float64[]
    for _ in 1:n_trials
        f0_tmp = copy(f0_fb)
        push!(times_fb, @elapsed execute_m2_scft!(m2fb, f0_tmp))
    end
    t_fb = minimum(times_fb)

    # Check fwd+bwd accuracy
    f0_fb_test = copy(f0_fb)
    execute_m2_scft!(m2fb, f0_fb_test)
    u_ref = fullgrid_diffuse(u_sym, N, Δs, lattice)
    f0_ref = extract_subgrid(u_ref, N, L)
    err_fb = maximum(abs.(f0_fb_test .- f0_ref))

    # ── M2+Q ──
    m2q = plan_m2_q(N, sg, 3, Δs, lattice)
    f0_q = extract_subgrid(u_sym, N, L)
    f0_q_3d = reshape(f0_q, Tuple(M))
    for _ in 1:n_warmup
        f0_tmp = copy(f0_q_3d)
        execute_m2_q!(m2q, f0_tmp)
    end
    times_q = Float64[]
    for _ in 1:n_trials
        f0_tmp = copy(f0_q_3d)
        push!(times_q, @elapsed execute_m2_q!(m2q, f0_tmp))
    end
    t_q = minimum(times_q)

    # Check M2+Q accuracy
    f0_q_test = copy(f0_q_3d)
    execute_m2_q!(m2q, f0_q_test)
    err_q = maximum(abs.(f0_q_test .- f0_ref))

    spd_fb = t_fft / t_fb
    spd_q = t_fft / t_q
    ratio = t_fb / t_q

    @printf("%-10s SG%-3d L=%-5s |G|=%-3d n_spec=%-5d  fft=%.2fms  fb=%.2fms(%4.1f×)  Q=%.2fms(%4.1f×)  fb/Q=%.2f  err_fb=%.1e  err_Q=%.1e\n",
            name, sg, L_str, length(ops_s), n_spec,
            t_fft*1e3, t_fb*1e3, spd_fb, t_q*1e3, spd_q, ratio,
            err_fb, err_q)

    return (name=name, sg=sg, L=L_str, nG=length(ops_s), n_spec=n_spec,
            N=N_size, t_fft=t_fft, t_fb=t_fb, t_q=t_q,
            spd_fb=spd_fb, spd_q=spd_q, ratio=ratio,
            err_fb=err_fb, err_q=err_q)
end

function main()
    FFTW.set_num_threads(1)

    println("=" ^130)
    println("M2 fwd+bwd vs M2+Q Benchmark (All space groups, single-threaded FFTW)")
    println("=" ^130)
    println()

    test_cases = [
        # Cubic P-centering
        (221, "Pm-3m"),
        (200, "Pm-3"),
        # Cubic centered
        (225, "Fm-3m"),
        (227, "Fd-3m"),
        (229, "Im-3m"),
        (230, "Ia-3d"),
        # Orthorhombic centered
        (70,  "Fddd"),
        (63,  "Cmcm"),
        (72,  "Ibam"),
        (74,  "Imma"),
        # Tetragonal
        (139, "I4/mmm"),
        (123, "P4/mmm"),
        # Lower symmetry
        (47,  "Pmmm"),
        (10,  "P2/m"),
    ]

    all_results = []

    for (N_size, n_trials) in [(64, 10), (128, 5)]
        println("--- N=$N_size ($n_trials trials, warmup=3) ---")
        @printf("%-10s %-5s %-7s %-5s %-7s  %-9s  %-16s  %-16s  %-6s  %-10s %-10s\n",
                "Group", "SG", "L", "|G|", "n_spec", "FFT(ms)",
                "fb(ms)(spdup)", "Q(ms)(spdup)", "fb/Q", "err_fb", "err_Q")
        println("-" ^130)

        u_tmp = complex(randn(N_size, N_size, N_size))
        fft_out = similar(u_tmp)
        fft_plan = plan_fft(u_tmp)

        for (sg, name) in test_cases
            r = benchmark_one(sg, name, N_size, fft_plan, fft_out;
                              n_warmup=3, n_trials=n_trials)
            push!(all_results, r)
        end
        println()
    end

    # Summary table
    println()
    println("=" ^100)
    println("Summary: fwd+bwd / Q Ratio and Accuracy")
    println("=" ^100)
    @printf("%-10s %-5s %-7s %10s %10s %12s %12s\n",
            "Group", "SG", "L", "fb/Q N=64", "fb/Q N=128", "err_fb", "err_Q")
    println("-" ^100)
    for (sg, name) in test_cases
        rs64 = filter(r -> r.sg == sg && r.N == 64, all_results)
        rs128 = filter(r -> r.sg == sg && r.N == 128, all_results)
        r64 = isempty(rs64) ? "—" : @sprintf("%.2f", rs64[1].ratio)
        r128 = isempty(rs128) ? "—" : @sprintf("%.2f", rs128[1].ratio)
        L_str = isempty(rs64) ? "?" : rs64[1].L
        efb = isempty(rs64) ? "—" : @sprintf("%.1e", rs64[1].err_fb)
        eq = isempty(rs64) ? "—" : @sprintf("%.1e", rs64[1].err_q)
        @printf("%-10s %-5d %-7s %10s %10s %12s %12s\n",
                name, sg, L_str, r64, r128, efb, eq)
    end
    println("=" ^100)
end

main()
