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

    # ── 9. Real CFFT (rcfft/ircfft) ──────────────────────────────────────
    @testset "Real CFFT (rcfft/ircfft)" begin

        @testset "Plan dispatch" begin
            fwd = plan_rcfft(N16, 221, 3)
            @test fwd isa RCFFTPlan

            # Centered groups also work
            fwd_f = plan_rcfft(N16, 225, 3)
            @test fwd_f isa RCFFTPlan
            fwd_i = plan_rcfft(N16, 229, 3)
            @test fwd_i isa RCFFTPlan
            @test subgrid_size(fwd) == (8, 8, 8)
            @test fullgrid_size(fwd) == N16
            @test stride_factors(fwd) == (2, 2, 2)
            @test cfft_asu_size(fwd) > 0

            bwd = plan_ircfft(fwd)
            @test bwd isa IRCFFTPlan
            @test subgrid_size(bwd) == (8, 8, 8)

            pair = plan_rcfft_pair(N16, 221, 3)
            @test pair isa GeneralRCFFTPairPlan
        end

        @testset "Roundtrip" begin
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
                    pair = plan_rcfft_pair(N16, sg, 3)
                    L = stride_factors(pair)
                    f0 = extract_subgrid(p.u, N16, collect(L))

                    F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
                    rcfft!(F̂, pair, f0)
                    f0_out = zeros(subgrid_size(pair)...)
                    ircfft!(f0_out, pair, F̂)

                    @test maximum(abs.(f0 .- f0_out)) < 1e-10
                end
            end
        end

        @testset "rcfft! ≈ cfft!" begin
            for (sg, name) in [(221, "Pm-3m"), (47, "Pmmm"), (2, "P-1"),
                               (225, "Fm-3m"), (229, "Im-3m")]
                @testset "$name (SG$sg)" begin
                    p = prep[sg]
                    pair_c = plan_cfft_pair(N16, sg, 3; method=:general)
                    pair_r = plan_rcfft_pair(N16, sg, 3)

                    L = stride_factors(pair_c)
                    f0 = extract_subgrid(p.u, N16, collect(L))

                    F̂_c = Vector{ComplexF64}(undef, cfft_asu_size(pair_c))
                    F̂_r = Vector{ComplexF64}(undef, cfft_asu_size(pair_r))
                    cfft!(F̂_c, pair_c, f0)
                    rcfft!(F̂_r, pair_r, f0)

                    @test maximum(abs.(F̂_c .- F̂_r)) < 1e-12
                end
            end
        end

        @testset "SCFT diffusion" begin
            for (sg, name) in [(221, "Pm-3m"), (47, "Pmmm"), (10, "P2/m"),
                               (225, "Fm-3m"), (229, "Im-3m")]
                @testset "$name (SG$sg)" begin
                    p = prep[sg]
                    pair = plan_rcfft_pair(N16, sg, 3)
                    L = stride_factors(pair)
                    f0 = extract_subgrid(p.u, N16, collect(L))

                    K = make_diffusion_kernel(pair, Δs, lattice)
                    F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
                    f0_new = copy(f0)
                    rcfft!(F̂, pair, f0_new)
                    @. F̂ *= K
                    ircfft!(f0_new, pair, F̂)

                    f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, collect(L))
                    @test maximum(abs.(f0_new .- f0_ref)) < 1e-10
                end
            end
        end
    end

    # ── 10. Star API ─────────────────────────────────────────────────────
    @testset "Star API" begin
        @testset "Structure (Pm-3m)" begin
            fwd = plan_cfft(N16, 221, 3)
            smap = build_subgrid_star_map(fwd)

            M = subgrid_size(fwd)
            M_vol = prod(M)
            @test length(smap.sub_to_star) == M_vol
            @test all(1 .<= smap.sub_to_star .<= smap.n_stars)
            @test length(smap.star_offsets) == smap.n_stars + 1
            @test length(smap.star_sub_list) == M_vol
            @test sum(smap.sub_degeneracy) == M_vol
            @test smap.n_stars > 0
            @test smap.n_stars <= M_vol
        end

        @testset "Expand/Compress roundtrip" begin
            fwd = plan_cfft(N16, 221, 3)
            smap = build_subgrid_star_map(fwd)
            M = subgrid_size(fwd)

            # Constant per star should roundtrip
            compressed = collect(Float64, 1:smap.n_stars)
            f0 = zeros(M...)
            expand_stars!(f0, smap, compressed)

            compressed_back = zeros(smap.n_stars)
            compress_stars!(compressed_back, smap, f0)
            @test maximum(abs.(compressed .- compressed_back)) < 1e-12
        end

        @testset "Symmetry invariance" begin
            for (sg, name) in [(221, "Pm-3m"), (225, "Fm-3m"),
                               (229, "Im-3m"), (47, "Pmmm")]
                @testset "$name" begin
                    p = prep[sg]
                    fwd = plan_cfft(N16, sg, 3)
                    smap = build_subgrid_star_map(fwd)
                    L = stride_factors(fwd)
                    f0 = extract_subgrid(p.u, N16, collect(L))

                    # For symmetric input, points in same star should be equal
                    for s in 1:smap.n_stars
                        idx_start = smap.star_offsets[s]
                        idx_end = smap.star_offsets[s + 1] - 1
                        vals = [f0[smap.star_sub_list[j]] for j in idx_start:idx_end]
                        @test maximum(abs.(vals .- vals[1])) < 1e-10
                    end
                end
            end
        end

        @testset "n_stars bounds" begin
            # For trivial group (P1), each subgrid point is its own star
            fwd_p1 = plan_cfft(N16, 1, 3)
            smap_p1 = build_subgrid_star_map(fwd_p1)
            @test smap_p1.n_stars == prod(subgrid_size(fwd_p1))

            # For high symmetry, n_stars << M_vol
            fwd_pm3m = plan_cfft(N16, 221, 3)
            smap_pm3m = build_subgrid_star_map(fwd_pm3m)
            @test smap_pm3m.n_stars < prod(subgrid_size(fwd_pm3m))
        end
    end

    # ── 11. cfft_kk_orbsum ────────────────────────────────────────────────
    @testset "cfft_kk_orbsum" begin
        @testset "Shape and Γ-point" begin
            fwd = plan_cfft(N16, 221, 3)
            kk = cfft_kk_orbsum(fwd, lattice)

            @test length(kk) == 6
            @test all(length(v) == cfft_asu_size(fwd) for v in kk)
            # Γ point (k=0): all components should be 0
            @test all(abs(kk[v][1]) < 1e-15 for v in 1:6)
        end

        @testset "Trace vs k²" begin
            fwd = plan_cfft(N16, 221, 3)
            kk = cfft_kk_orbsum(fwd, lattice)
            k2 = cfft_k2(fwd, lattice)

            # Trace kk_xx + kk_yy + kk_zz should equal k²
            trace = kk[1] .+ kk[2] .+ kk[3]
            @test maximum(abs.(trace .- k2)) < 1e-10
        end

        @testset "Cubic symmetry" begin
            # For cubic lattice, orbit-averaged xx ≈ yy ≈ zz
            fwd = plan_cfft(N16, 221, 3)
            kk = cfft_kk_orbsum(fwd, lattice)

            # Skip Γ point
            for i in 2:cfft_asu_size(fwd)
                @test abs(kk[1][i] - kk[2][i]) < 1e-10
                @test abs(kk[2][i] - kk[3][i]) < 1e-10
            end
        end

        @testset "Multi-group" begin
            for (sg, name) in [(225, "Fm-3m"), (47, "Pmmm"), (2, "P-1")]
                @testset "$name" begin
                    fwd = plan_cfft(N16, sg, 3)
                    kk = cfft_kk_orbsum(fwd, lattice)
                    k2 = cfft_k2(fwd, lattice)

                    @test length(kk) == 6
                    trace = kk[1] .+ kk[2] .+ kk[3]
                    @test maximum(abs.(trace .- k2)) < 1e-10
                end
            end
        end
    end

end
