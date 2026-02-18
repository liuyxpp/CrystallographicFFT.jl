# ============================================================================
# CrystallographicFFT.jl — Test Suite (Device-Agnostic API)
#
# Coverage: ASU, SpectralIndexing, new CFFT API (general + centered)
# Target:   < 2 min
# Strategy: representative groups per centering, N=16³, shared helpers
# ============================================================================

using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp, apply_op,
    check_shift_invariance, detect_centering_type, CentF, CentI, CentC, CentP
using CrystallographicFFT.ASU: find_optimal_shift, calc_asu, pack_asu
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.CFFTApi
using FFTW
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "CrystallographicFFT.jl" begin

    # ── 1. ASU Construction + Shift ──────────────────────────────────────
    @testset "ASU Construction" begin
        @testset "Shift search (p2mm)" begin
            N = (8, 8)
            ops = get_ops(6, 2, N)
            shift, shifted_ops = find_optimal_shift(ops, N)
            @test shift ≈ [0.5/8, 0.5/8]
            valid, _ = check_shift_invariance(ops, collect(shift), N)
            @test valid
        end

        @testset "Coverage & disjointness" begin
            for (sg, dim, N, expected) in [
                (6, 2, (8,8), 16),
                (47, 3, (8,8,8), 64),
            ]
                points, shift = calc_asu(sg, dim, N)
                @test length(points) == expected
                _, shifted_ops = find_optimal_shift(get_ops(sg, dim, N), N)
                coverage = Set{Vector{Int}}()
                for p in points
                    orb = compute_full_orbit(p.idx, shifted_ops, N)
                    @test isempty(intersect(coverage, orb))
                    union!(coverage, orb)
                end
                @test length(coverage) == prod(N)
            end
        end
    end

    # ── 2. ASU Packing ───────────────────────────────────────────────────
    @testset "ASU Packing" begin
        for (sg, dim, N) in [(6, 2, (8,8)), (47, 3, (8,8,8))]
            @testset "SG $sg" begin
                points, shift = calc_asu(sg, dim, N)
                c = pack_asu(points, N; shift=shift)
                total = sum(length(b.data) for (_, blocks) in c.dim_blocks for b in blocks)
                @test total == length(points)
            end
        end
    end

    # ── Common setup for 3D tests ────────────────────────────────────────
    N16 = (16, 16, 16)
    lattice = Matrix{Float64}(I, 3, 3)
    Δs = 0.05

    # Pre-compute frequently-used shifted ops + make_symmetric
    prep = Dict{Int, NamedTuple}()
    for sg in [221, 225, 229, 63, 47, 227, 230, 70, 72, 10, 2, 123]
        ops = get_ops(sg, 3, N16)
        _, ops_s = find_optimal_shift(ops, N16)
        spec = calc_spectral_asu(ops_s, 3, N16)
        u = make_symmetric(ops_s, N16)
        prep[sg] = (; ops, ops_s, spec, u)
    end

    # ── 3. Spectral ASU ──────────────────────────────────────────────────
    @testset "Spectral ASU" begin
        @testset "spec vs full FFT (Pm-3m)" begin
            p = prep[221]
            ref = fft(ComplexF64.(p.u))
            for i in 1:min(50, length(p.spec.points))
                h = get_k_vector(p.spec, i)
                ci = CartesianIndex(Tuple(mod.(h, N16) .+ 1))
                # ASU should produce valid h-vectors within N
                @test all(0 .<= h .< N16[1])
            end
        end
    end

    # ── 4. CFFT API: plan_cfft_pair dispatch ─────────────────────────────
    @testset "CFFT API" begin
        @testset "plan_cfft_pair dispatch" begin
            plan_g = plan_cfft_pair(N16, 221, 3)
            @test plan_g isa GeneralCFFTPairPlan
            @test subgrid_size(plan_g) == (8, 8, 8)
            @test fullgrid_size(plan_g) == N16
            @test stride_factors(plan_g) == (2, 2, 2)
            @test cfft_asu_size(plan_g) > 0

            plan_c = plan_cfft_pair(N16, 225, 3)
            @test plan_c isa CenteredCFFTPairPlan
            @test subgrid_size(plan_c) == (8, 8, 8)

            # Force general path for centered group
            plan_c_gen = plan_cfft_pair(N16, 225, 3; method=:general)
            @test plan_c_gen isa GeneralCFFTPairPlan
        end

        # ── plan_cfft / plan_icfft (single-direction) ──
        @testset "Single-direction plans" begin
            fwd = plan_cfft(N16, 221, 3)
            @test fwd isa CFFTPlan
            @test subgrid_size(fwd) == (8, 8, 8)

            bwd = plan_icfft(fwd)
            @test bwd isa ICFFTPlan
            @test subgrid_size(bwd) == (8, 8, 8)

            # Standalone backward
            bwd2 = plan_icfft(N16, 221, 3)
            @test bwd2 isa ICFFTPlan
        end

        # ── cfft!/icfft! roundtrip (PairPlan, General) ──
        @testset "Roundtrip General (Pm-3m)" begin
            plan_g = plan_cfft_pair(N16, 221, 3)
            p = prep[221]
            L = stride_factors(plan_g)
            M = subgrid_size(plan_g)
            f0 = extract_subgrid(p.u, N16, collect(L))

            F̂ = Vector{ComplexF64}(undef, cfft_asu_size(plan_g))
            cfft!(F̂, plan_g, f0)
            f0_out = zeros(M...)
            icfft!(f0_out, plan_g, F̂)

            @test maximum(abs.(f0 .- f0_out)) < 1e-10
        end

        # ── cfft!/icfft! roundtrip (PairPlan, Centered) ──
        @testset "Roundtrip Centered (Fm-3m)" begin
            plan_c = plan_cfft_pair(N16, 225, 3)
            p = prep[225]
            f0 = extract_subgrid(p.u, N16, [2, 2, 2])

            F̂ = Vector{ComplexF64}(undef, cfft_asu_size(plan_c))
            cfft!(F̂, plan_c, f0)
            f0_out = zeros(subgrid_size(plan_c)...)
            icfft!(f0_out, plan_c, F̂)

            @test maximum(abs.(f0 .- f0_out)) < 1e-10
        end

        # ── cfft!/icfft! roundtrip (Single-direction CFFTPlan + ICFFTPlan) ──
        @testset "Roundtrip CFFTPlan+ICFFTPlan (Pm-3m)" begin
            fwd = plan_cfft(N16, 221, 3)
            bwd = plan_icfft(fwd)
            p = prep[221]
            f0 = extract_subgrid(p.u, N16, collect(stride_factors(fwd)))

            F̂ = Vector{ComplexF64}(undef, cfft_asu_size(fwd))
            cfft!(F̂, fwd, f0)
            f0_out = zeros(subgrid_size(bwd)...)
            icfft!(f0_out, bwd, F̂)

            @test maximum(abs.(f0 .- f0_out)) < 1e-10
        end

        # ── Diffusion kernel vs FFT reference ──
        @testset "Diffusion kernel (Pm-3m)" begin
            plan_g = plan_cfft_pair(N16, 221, 3)
            p = prep[221]
            f0 = extract_subgrid(p.u, N16, collect(stride_factors(plan_g)))

            K = make_diffusion_kernel(plan_g, Δs, lattice)
            @test length(K) == cfft_asu_size(plan_g)
            @test all(0 .< K .<= 1)

            F̂ = Vector{ComplexF64}(undef, cfft_asu_size(plan_g))
            f0_new = copy(f0)
            cfft!(F̂, plan_g, f0_new)
            @. F̂ *= K
            icfft!(f0_new, plan_g, F̂)

            # Compare with full-grid FFT reference
            f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, [2,2,2])
            @test maximum(abs.(f0_new .- f0_ref)) < 1e-10
        end

        # ── update_diffusion_kernel! ──
        @testset "update_diffusion_kernel!" begin
            fwd = plan_cfft(N16, 221, 3)
            K = make_diffusion_kernel(fwd, Δs, lattice)
            update_diffusion_kernel!(K, fwd, 0.10, lattice)
            K_ref = make_diffusion_kernel(fwd, 0.10, lattice)

            @test maximum(abs.(K .- K_ref)) < 1e-15
        end

        # ── cfft_k2 ──
        @testset "cfft_k2" begin
            fwd = plan_cfft(N16, 221, 3)
            k2 = cfft_k2(fwd, lattice)
            @test length(k2) == cfft_asu_size(fwd)
            @test all(k2 .>= 0)
            @test k2[1] ≈ 0.0  # Γ point
        end

        # ── Grid conversions ──
        @testset "Grid conversions" begin
            fwd = plan_cfft(N16, 221, 3)
            p = prep[221]
            f0 = extract_subgrid(p.u, N16, collect(stride_factors(fwd)))

            # subgrid → fullgrid → extract back = original
            f_full = zeros(N16...)
            CrystallographicFFT.CFFTApi.subgrid_to_fullgrid!(f_full, fwd, f0)
            @test maximum(abs.(f_full .- p.u)) < 1e-12

            f0_back = zeros(subgrid_size(fwd)...)
            CrystallographicFFT.CFFTApi.fullgrid_to_subgrid!(f0_back, fwd, p.u)
            @test maximum(abs.(f0 .- f0_back)) < 1e-12
        end
    end

    # ── 5. Multi-group spectral consistency ──────────────────────────────
    @testset "Spectral Consistency" begin
        for (sg, name, expected_type) in [
            (221, "Pm-3m",  :general),
            (225, "Fm-3m",  :centered),
            (229, "Im-3m",  :centered),
            (63,  "Cmcm",   :centered),
            (70,  "Fddd",   :centered),
            (47,  "Pmmm",   :general),
            (2,   "P-1",    :general),
            (123, "P4/mmm", :general),
        ]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                fwd = plan_cfft(N16, sg, 3)
                M = subgrid_size(fwd)
                f0 = extract_subgrid(p.u, N16, collect(stride_factors(fwd)))

                F̂ = Vector{ComplexF64}(undef, cfft_asu_size(fwd))
                cfft!(F̂, fwd, f0)

                F_ref = fft(ComplexF64.(p.u))
                max_err = 0.0
                for i in 1:min(100, cfft_asu_size(fwd))
                    h = get_k_vector(fwd.spec_asu, i)
                    ci = CartesianIndex(Tuple(mod.(h, N16) .+ 1))
                    max_err = max(max_err, abs(F̂[i] - F_ref[ci]))
                end
                @test max_err < 1e-8
            end
        end
    end

    # ── 6. Multi-group roundtrip ─────────────────────────────────────────
    @testset "Roundtrip Multi-Group" begin
        for (sg, name) in [
            (221, "Pm-3m"),
            (225, "Fm-3m"),
            (229, "Im-3m"),
            (63,  "Cmcm"),
            (70,  "Fddd"),
            (227, "Fd-3m"),
            (230, "Ia-3d"),
            (72,  "Ibam"),
            (10,  "P2/m"),
            (2,   "P-1"),
            (123, "P4/mmm"),
        ]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                pair = plan_cfft_pair(N16, sg, 3)
                M = subgrid_size(pair)
                L = stride_factors(pair)
                f0 = extract_subgrid(p.u, N16, collect(L))

                F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
                cfft!(F̂, pair, f0)
                f0_out = zeros(M...)
                icfft!(f0_out, pair, F̂)

                @test maximum(abs.(f0 .- f0_out)) < 1e-10
            end
        end
    end

    # ── 7. SCFT diffusion (fwd + kernel + bwd) vs full-grid FFT ──────────
    @testset "SCFT Diffusion" begin
        for (sg, name) in [
            (221, "Pm-3m"),
            (225, "Fm-3m"),
            (229, "Im-3m"),
            (70,  "Fddd"),
            (10,  "P2/m"),
        ]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                pair = plan_cfft_pair(N16, sg, 3)
                L = stride_factors(pair)
                f0 = extract_subgrid(p.u, N16, collect(L))
                f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, collect(L))

                K = make_diffusion_kernel(pair, Δs, lattice)
                F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
                f0_new = copy(f0)
                cfft!(F̂, pair, f0_new)
                @. F̂ *= K
                icfft!(f0_new, pair, F̂)

                @test maximum(abs.(f0_new .- f0_ref)) < 1e-10
            end
        end
    end

    # ── 8. Centering detection ───────────────────────────────────────────
    @testset "Centering Detection" begin
        for (sg, expected_cent) in [
            (221, CentP),  # Pm-3m
            (225, CentF),  # Fm-3m
            (229, CentI),  # Im-3m
            (63,  CentC),  # Cmcm
            (2,   CentP),  # P-1
        ]
            p = prep[sg]
            cent = detect_centering_type(p.ops_s, N16)
            @test cent == expected_cent
        end
    end

end
