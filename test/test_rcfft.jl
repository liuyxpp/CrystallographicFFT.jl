using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.CFFTApi
using FFTW
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Real CFFT (rcfft/ircfft)" begin

    N = (16, 16, 16)
    lattice = Matrix{Float64}(I, 3, 3)
    Δs = 0.05

    # Pre-compute shifted ops for all test groups (general + centered)
    prep = Dict{Int, NamedTuple}()
    for sg in [221, 47, 2, 123, 10, 225, 229]
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u = make_symmetric(ops_s, N)
        prep[sg] = (; ops, ops_s, spec, u)
    end

    # ── 1. Plan dispatch ────────────────────────────────────────────────
    @testset "Plan dispatch" begin
        fwd = plan_rcfft(N, 221, 3)
        @test fwd isa RCFFTPlan
        @test subgrid_size(fwd) == (8, 8, 8)
        @test fullgrid_size(fwd) == N
        @test stride_factors(fwd) == (2, 2, 2)
        @test cfft_asu_size(fwd) > 0

        bwd = plan_ircfft(fwd)
        @test bwd isa IRCFFTPlan
        @test subgrid_size(bwd) == (8, 8, 8)

        # Standalone backward
        bwd2 = plan_ircfft(N, 221, 3)
        @test bwd2 isa IRCFFTPlan

        # Pair plan
        pair = plan_rcfft_pair(N, 221, 3)
        @test pair isa GeneralRCFFTPairPlan
        @test subgrid_size(pair) == (8, 8, 8)
    end

    # ── 2. Roundtrip (rcfft! → ircfft! ≈ identity) ─────────────────────
    @testset "Roundtrip" begin
        for (sg, name) in [
            (221, "Pm-3m"),
            (47,  "Pmmm"),
            (2,   "P-1"),
            (123, "P4/mmm"),
            (10,  "P2/m"),
            (225, "Fm-3m"),
            (229, "Im-3m"),
        ]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                pair = plan_rcfft_pair(N, sg, 3)
                L = stride_factors(pair)
                M = subgrid_size(pair)
                f0 = extract_subgrid(p.u, N, collect(L))

                F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
                rcfft!(F̂, pair, f0)
                f0_out = zeros(M...)
                ircfft!(f0_out, pair, F̂)

                @test maximum(abs.(f0 .- f0_out)) < 1e-10
            end
        end
    end

    # ── 3. Spectral consistency (rcfft! vs full-grid FFT) ──────────────
    @testset "Spectral consistency" begin
        for (sg, name) in [
            (221, "Pm-3m"),
            (47,  "Pmmm"),
            (2,   "P-1"),
            (123, "P4/mmm"),
            (225, "Fm-3m"),
            (229, "Im-3m"),
        ]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                fwd = plan_rcfft(N, sg, 3)
                L = stride_factors(fwd)
                M = subgrid_size(fwd)
                f0 = extract_subgrid(p.u, N, collect(L))

                F̂ = Vector{ComplexF64}(undef, cfft_asu_size(fwd))
                rcfft!(F̂, fwd, f0)

                F_ref = fft(ComplexF64.(p.u))
                max_err = 0.0
                for i in 1:min(100, cfft_asu_size(fwd))
                    h = get_k_vector(fwd.spec_asu, i)
                    ci = CartesianIndex(Tuple(mod.(h, N) .+ 1))
                    max_err = max(max_err, abs(F̂[i] - F_ref[ci]))
                end
                @test max_err < 1e-8
            end
        end
    end

    # ── 4. Cross-validation: rcfft! ≈ cfft! ────────────────────────────
    @testset "rcfft! ≈ cfft!" begin
        for (sg, name) in [
            (221, "Pm-3m"),
            (47,  "Pmmm"),
            (2,   "P-1"),
            (225, "Fm-3m"),
            (229, "Im-3m"),
        ]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                pair_c = plan_cfft_pair(N, sg, 3; method=:general)
                pair_r = plan_rcfft_pair(N, sg, 3)

                L = stride_factors(pair_c)
                f0 = extract_subgrid(p.u, N, collect(L))

                F̂_c = Vector{ComplexF64}(undef, cfft_asu_size(pair_c))
                F̂_r = Vector{ComplexF64}(undef, cfft_asu_size(pair_r))
                cfft!(F̂_c, pair_c, f0)
                rcfft!(F̂_r, pair_r, f0)

                @test maximum(abs.(F̂_c .- F̂_r)) < 1e-12
            end
        end
    end

    # ── 5. SCFT diffusion (rcfft! → K·F̂ → ircfft!) ────────────────────
    @testset "SCFT diffusion" begin
        for (sg, name) in [
            (221, "Pm-3m"),
            (47,  "Pmmm"),
            (10,  "P2/m"),
            (225, "Fm-3m"),
            (229, "Im-3m"),
        ]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                pair = plan_rcfft_pair(N, sg, 3)
                L = stride_factors(pair)
                f0 = extract_subgrid(p.u, N, collect(L))

                K = make_diffusion_kernel(pair, Δs, lattice)
                @test length(K) == cfft_asu_size(pair)
                @test all(0 .< K .<= 1)

                F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
                f0_new = copy(f0)
                rcfft!(F̂, pair, f0_new)
                @. F̂ *= K
                ircfft!(f0_new, pair, F̂)

                f0_ref = fullgrid_reference(p.u, N, Δs, lattice, collect(L))
                @test maximum(abs.(f0_new .- f0_ref)) < 1e-10
            end
        end
    end

    # ── 6. Single-direction plan roundtrip ─────────────────────────────
    @testset "Roundtrip RCFFTPlan+IRCFFTPlan" begin
        fwd = plan_rcfft(N, 221, 3)
        bwd = plan_ircfft(fwd)
        p = prep[221]
        f0 = extract_subgrid(p.u, N, collect(stride_factors(fwd)))

        F̂ = Vector{ComplexF64}(undef, cfft_asu_size(fwd))
        rcfft!(F̂, fwd, f0)
        f0_out = zeros(subgrid_size(bwd)...)
        ircfft!(f0_out, bwd, F̂)

        @test maximum(abs.(f0 .- f0_out)) < 1e-10
    end
end
