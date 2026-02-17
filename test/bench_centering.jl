#=
Centering Packing Feasibility Benchmark
========================================
Evaluate whether KRFFT III centering packing (χ-weighted linear combination)
can be cost-effective vs the FFT savings it provides.

Two centering types tested:
- I-centering (Im-3m SG229): 1 centering vector → 2× FFT reduction
- F-centering (Fm-3m SG225): 3 centering vectors → 4× FFT reduction

The benchmark measures:
1. packing overhead: cost of χ-weighted sum vs simple stride-2 gather
2. FFT savings: time saved by eliminating extinct sub-FFTs
3. Net benefit: whether packing overhead < FFT savings
=#

using FFTW
using BenchmarkTools
using Printf

FFTW.set_num_threads(1)

# ─── Centering configurations ───

# I-centering: τ = (N/2, N/2, N/2), 2 alive / 8 total sectors → 4 alive
const I_CENT_VECS = [[1, 1, 1]]  # in units of N/2

# F-centering: τ₁,τ₂,τ₃, 2 alive / 8 total sectors → 2 alive
const F_CENT_VECS = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]  # in units of N/2

"""
Compute which parity sectors survive centering extinction.

For centering group G_c with translations τ_i, sector p survives iff:
  |Σ_{τ∈G_c} exp(-2πi p·τ/N)| > 0.5

where p ∈ {0,1}^3 and τ is in grid coordinates (τ_actual = cent_vec * N/2).
"""
function alive_sectors(cent_vecs::Vector{Vector{Int}})
    # Build centering group (closure under addition mod 2)
    group = Set{Vector{Int}}()
    push!(group, [0, 0, 0])
    queue = [[0, 0, 0]]
    for v in cent_vecs
        if v ∉ group
            push!(group, v)
            push!(queue, v)
        end
    end
    while !isempty(queue)
        g = pop!(queue)
        for v in cent_vecs
            h = mod.(g .+ v, 2)
            if h ∉ group
                push!(group, h)
                push!(queue, h)
            end
        end
    end

    alive = Int[]
    for bits in 0:7
        p = [(bits >> k) & 1 for k in 0:2]
        phase_sum = sum(cispi(-2.0 * sum(p .* τ) / 2.0) for τ in group)
        if abs(phase_sum) > 0.5
            push!(alive, bits)
        end
    end
    return alive, length(group)
end

# ─── Packing benchmarks ───

"""
Standard stride-2 packing: u_p(n) = u(2n + p)
This is what we currently do for all sectors.
"""
function pack_stride2!(buf::Array{ComplexF64,3}, u::Array{Float64,3}, p::NTuple{3,Int})
    M = size(buf)
    @inbounds for k in 1:M[3], j in 1:M[2], i in 1:M[1]
        ii = 2*(i-1) + p[1] + 1
        jj = 2*(j-1) + p[2] + 1
        kk = 2*(k-1) + p[3] + 1
        buf[i,j,k] = complex(u[ii, jj, kk])
    end
end

"""
KRFFT III centering packing: χ-weighted linear combination.

f_j(x) = (1/|G_c|) Σ_{τ∈G_c} χ_j(τ) · u(2x + p + τ·N/2)

where χ_j(τ) = exp(-2πi p_j·τ/2), p_j is the alive parity.
Since χ values are ±1 or ±i for half-grid centering, this reduces to
real additions/subtractions in practice.
"""
function pack_centering!(buf::Array{ComplexF64,3}, u::Array{Float64,3},
                         p::NTuple{3,Int}, cent_group::Vector{Vector{Int}},
                         N::NTuple{3,Int})
    M = size(buf)
    n_cent = length(cent_group)
    inv_n = 1.0 / n_cent

    # Precompute χ values: χ(τ) = exp(-2πi p·τ/2)
    chi = [cispi(-2.0 * sum(p[d] * cent_group[c][d] for d in 1:3) / 2.0)
           for c in 1:n_cent]

    @inbounds for k in 1:M[3], j in 1:M[2], i in 1:M[1]
        val = zero(ComplexF64)
        for c in 1:n_cent
            τ = cent_group[c]
            ii = mod(2*(i-1) + p[1] + τ[1] * (N[1] ÷ 2), N[1]) + 1
            jj = mod(2*(j-1) + p[2] + τ[2] * (N[2] ÷ 2), N[2]) + 1
            kk = mod(2*(k-1) + p[3] + τ[3] * (N[3] ÷ 2), N[3]) + 1
            val += chi[c] * u[ii, jj, kk]
        end
        buf[i,j,k] = val * inv_n
    end
end

"""
Optimized centering packing with precomputed gather table.
This is the version that would actually be used in production.
"""
function pack_centering_precomp!(buf::Vector{ComplexF64}, u_flat::Vector{Float64},
                                 src_tables::Vector{Vector{Int32}},
                                 chi::Vector{ComplexF64}, inv_n::Float64)
    n_cent = length(src_tables)
    M = length(buf)

    fill!(buf, zero(ComplexF64))
    for c in 1:n_cent
        src = src_tables[c]
        w = chi[c] * inv_n
        @inbounds @simd for i in 1:M
            buf[i] += w * u_flat[src[i]]
        end
    end
end

function build_centering_gather_tables(M::NTuple{3,Int}, p::NTuple{3,Int},
                                        cent_group::Vector{Vector{Int}},
                                        N::NTuple{3,Int})
    vol = prod(M)
    tables = Vector{Vector{Int32}}(undef, length(cent_group))
    for (c, τ) in enumerate(cent_group)
        tbl = Vector{Int32}(undef, vol)
        idx = 0
        for k in 1:M[3], j in 1:M[2], i in 1:M[1]
            idx += 1
            ii = mod(2*(i-1) + p[1] + τ[1] * (N[1] ÷ 2), N[1]) + 1
            jj = mod(2*(j-1) + p[2] + τ[2] * (N[2] ÷ 2), N[2]) + 1
            kk = mod(2*(k-1) + p[3] + τ[3] * (N[3] ÷ 2), N[3]) + 1
            tbl[idx] = Int32((kk-1) * N[1]*N[2] + (jj-1) * N[1] + ii)
        end
        tables[c] = tbl
    end
    return tables
end

# ─── Main benchmark ───

function run_benchmark(N_val::Int)
    N = (N_val, N_val, N_val)
    M = N .÷ 2

    u = rand(Float64, N...)
    u_flat = vec(u)

    println("=" ^ 70)
    @printf("N = %d,  M = (%d,%d,%d),  sub-FFT volume = %d\n",
            N_val, M..., prod(M))
    println("=" ^ 70)

    # 1. Baseline: full FFT
    u_c = complex(u)
    fft_plan = plan_fft(u_c)
    t_full_fft = @belapsed ($fft_plan * $u_c) samples=5 evals=3
    @printf("  Full FFT (%d³):        %8.3f ms\n", N_val, t_full_fft * 1000)

    # 2. Sub-grid FFT
    buf_sub = zeros(ComplexF64, M)
    fft_plan_sub = plan_fft(buf_sub)
    t_sub_fft = @belapsed ($fft_plan_sub * $buf_sub) samples=5 evals=3
    @printf("  Sub-FFT (%d³):         %8.3f ms\n", M[1], t_sub_fft * 1000)

    # 3. Standard pack (1 sector)
    buf = zeros(ComplexF64, M)
    p0 = (0, 0, 0)
    t_std_pack = @belapsed pack_stride2!($buf, $u, $p0) samples=5 evals=5
    @printf("  Std pack (1 sector):   %8.3f ms\n", t_std_pack * 1000)

    println()
    println("─── I-centering (1 centering vector, τ=(½,½,½)) ───")

    alive_I, ng_I = alive_sectors(I_CENT_VECS)
    println("  Centering group size: $ng_I,  alive sectors: $(length(alive_I))/8")

    # Build I-centering group
    icent_group = [[0,0,0], [1,1,1]]

    # Naive centering pack
    t_cent_naive_I = @belapsed pack_centering!($buf, $u, $p0, $icent_group, $N) samples=5 evals=3
    @printf("  Centering pack naive:  %8.3f ms  (%.1f× std pack)\n",
            t_cent_naive_I * 1000, t_cent_naive_I / t_std_pack)

    # Precomputed centering pack
    tables_I = build_centering_gather_tables(M, p0, icent_group, N)
    chi_I = [cispi(-2.0 * sum(p0[d] * icent_group[c][d] for d in 1:3) / 2.0)
             for c in 1:length(icent_group)]
    buf_flat = zeros(ComplexF64, prod(M))
    t_cent_precomp_I = @belapsed pack_centering_precomp!($buf_flat, $u_flat,
                                                          $tables_I, $chi_I,
                                                          $(1.0/ng_I)) samples=5 evals=5
    @printf("  Centering pack precomp:%8.3f ms  (%.1f× std pack)\n",
            t_cent_precomp_I * 1000, t_cent_precomp_I / t_std_pack)

    # Cost-benefit analysis
    n_eliminated_I = 8 - length(alive_I)
    fft_saving_I = n_eliminated_I * t_sub_fft
    pack_overhead_I = length(alive_I) * (t_cent_precomp_I - t_std_pack)
    net_I = fft_saving_I - pack_overhead_I

    @printf("\n  FFT savings:    %d eliminated × %.3f ms = %.3f ms\n",
            n_eliminated_I, t_sub_fft * 1000, fft_saving_I * 1000)
    @printf("  Pack overhead:  %d alive × (%.3f - %.3f) ms = %.3f ms\n",
            length(alive_I), t_cent_precomp_I * 1000, t_std_pack * 1000,
            pack_overhead_I * 1000)
    @printf("  ➜ Net benefit:  %.3f ms  (%s)\n",
            net_I * 1000, net_I > 0 ? "✅ WORTH IT" : "❌ NOT WORTH IT")

    println()
    println("─── F-centering (3 centering vectors) ───")

    alive_F, ng_F = alive_sectors(F_CENT_VECS)
    println("  Centering group size: $ng_F,  alive sectors: $(length(alive_F))/8")

    # Build F-centering group
    fcent_group = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]

    # Naive centering pack
    t_cent_naive_F = @belapsed pack_centering!($buf, $u, $p0, $fcent_group, $N) samples=5 evals=3
    @printf("  Centering pack naive:  %8.3f ms  (%.1f× std pack)\n",
            t_cent_naive_F * 1000, t_cent_naive_F / t_std_pack)

    # Precomputed centering pack
    tables_F = build_centering_gather_tables(M, p0, fcent_group, N)
    chi_F = [cispi(-2.0 * sum(p0[d] * fcent_group[c][d] for d in 1:3) / 2.0)
             for c in 1:length(fcent_group)]
    t_cent_precomp_F = @belapsed pack_centering_precomp!($buf_flat, $u_flat,
                                                          $tables_F, $chi_F,
                                                          $(1.0/ng_F)) samples=5 evals=5
    @printf("  Centering pack precomp:%8.3f ms  (%.1f× std pack)\n",
            t_cent_precomp_F * 1000, t_cent_precomp_F / t_std_pack)

    # Cost-benefit analysis
    n_eliminated_F = 8 - length(alive_F)
    fft_saving_F = n_eliminated_F * t_sub_fft
    pack_overhead_F = length(alive_F) * (t_cent_precomp_F - t_std_pack)
    net_F = fft_saving_F - pack_overhead_F

    @printf("\n  FFT savings:    %d eliminated × %.3f ms = %.3f ms\n",
            n_eliminated_F, t_sub_fft * 1000, fft_saving_F * 1000)
    @printf("  Pack overhead:  %d alive × (%.3f - %.3f) ms = %.3f ms\n",
            length(alive_F), t_cent_precomp_F * 1000, t_std_pack * 1000,
            pack_overhead_F * 1000)
    @printf("  ➜ Net benefit:  %.3f ms  (%s)\n",
            net_F * 1000, net_F > 0 ? "✅ WORTH IT" : "❌ NOT WORTH IT")

    println()
    println("─── Context: current G0 ASU pipeline ───")
    # 4 sub-FFTs at (N/4)³
    M4 = N .÷ 4
    buf4 = zeros(ComplexF64, M4)
    fft_plan_4 = plan_fft(buf4)
    t_sub4_fft = @belapsed ($fft_plan_4 * $buf4) samples=5 evals=3
    @printf("  Sub-FFT (%d³):         %8.3f ms\n", M4[1], t_sub4_fft * 1000)
    @printf("  4 × sub-FFT (%d³):     %8.3f ms  (current Cubic G0 ASU FFT cost)\n",
            M4[1], 4 * t_sub4_fft * 1000)

    # With centering applied at P3c level (reduce 4 → 1 or 2 FFTs)
    if length(alive_F) <= 4
        println("\n  If centering reduces 4 P3c sub-FFTs to $(length(alive_F) ÷ 2):")
        n_saved = 4 - length(alive_F) ÷ 2
        @printf("    FFT savings: %d × %.3f ms = %.3f ms\n",
                n_saved, t_sub4_fft * 1000, n_saved * t_sub4_fft * 1000)
    end

    println()
    return (;N=N_val, t_full_fft, t_sub_fft, t_std_pack,
             t_cent_precomp_I, t_cent_precomp_F,
             net_I, net_F)
end

# Run for multiple grid sizes
println("\nCentering Packing Feasibility Benchmark")
println("FFTW threads: $(FFTW.get_num_threads())")
println()

results = []
for N in [32, 64, 128, 256]
    r = run_benchmark(N)
    push!(results, r)
end

# Summary table
println("\n" * "=" ^ 70)
println("SUMMARY TABLE")
println("=" ^ 70)
@printf("%5s  %10s  %10s  %8s  %10s  %10s  %8s\n",
        "N", "I-net(ms)", "I-viable?", "I-ratio",
        "F-net(ms)", "F-viable?", "F-ratio")
println("-" ^ 70)
for r in results
    i_ratio = r.net_I > 0 ? r.net_I / r.t_full_fft : 0.0
    f_ratio = r.net_F > 0 ? r.net_F / r.t_full_fft : 0.0
    @printf("%5d  %10.3f  %10s  %7.1f%%  %10.3f  %10s  %7.1f%%\n",
            r.N,
            r.net_I * 1000, r.net_I > 0 ? "✅ YES" : "❌ NO", i_ratio * 100,
            r.net_F * 1000, r.net_F > 0 ? "✅ YES" : "❌ NO", f_ratio * 100)
end
