# ============================================================================
# CrystallographicFFT.jl — Streamlined Test Suite
#
# Coverage: all modules via representative tests + tricky edge-case groups
# Target:   < 5 min (vs 86 min with full suite)
# Strategy: one representative group per centering type, N=16³, shared helpers
# Tricky groups (from docs): Fd-3m(227), Ia-3d(230), Fddd(70), Ibam(72), P2/m(10)
#
# For comprehensive regression: julia --project=test test/full/run_all.jl
# ============================================================================

using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp, apply_op,
    check_shift_invariance, detect_centering_type, CentF, CentI, CentC
using CrystallographicFFT.ASU: find_optimal_shift, calc_asu, pack_asu
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT: plan_krfft_selective, execute_selective_krfft!,
    plan_krfft_g0asu, execute_g0asu_krfft!,
    plan_krfft_g0asu_backward, execute_g0asu_ikrfft!,
    plan_krfft_centered, execute_centered_krfft!,
    plan_centered_ikrfft, execute_centered_ikrfft!,
    plan_centering_fold, centering_fold!, fft_channels!, assemble_G0!,
    ifft_channels!, centering_unfold!, disassemble_G0!,
    pack_stride_real!,
    plan_centered_scft, execute_centered_scft!, auto_L,
    SubgridCenteringFoldPlan, CenteredKRFFTPlan, CenteredKRFFTBackwardPlan
using CrystallographicFFT.QFusedKRFFT: plan_m2_q, execute_m2_q!,
    M2QPlan, fullgrid_to_subgrid!, subgrid_to_fullgrid!,
    plan_m7_scft, execute_m7_scft!,
    plan_m2_scft, execute_m2_scft!, update_m2_kernel!, M2SCFTPlan
using FFTW
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "CrystallographicFFT.jl" begin

    # ── 1. ASU Construction + Shift (2D + 3D) ────────────────────────────
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

    # ── 3. CFFT Plan + Roundtrip (2D) ────────────────────────────────────
    @testset "CFFT Roundtrip (p2mm)" begin
        sg_num = 6; N = (16, 16)
        plan = plan_cfft(N, sg_num, ComplexF64, Array)
        @test plan isa CFFTPlan

        input_asu = deepcopy(plan.asu)
        for (_, blocks) in input_asu.dim_blocks
            for b in blocks; b.data .= rand(ComplexF64, size(b.data)); end
        end

        spectral = deepcopy(plan.asu)
        mul!(spectral, plan, input_asu)

        recon = deepcopy(plan.asu)
        ldiv!(recon, plan, spectral)

        max_err = 0.0
        for (d, blocks) in recon.dim_blocks
            for (i, b) in enumerate(blocks)
                max_err = max(max_err, norm(b.data - input_asu.dim_blocks[d][i].data))
            end
        end
        @test max_err < 1e-10
    end

    # ── Common setup for 3D tests ────────────────────────────────────────
    N16 = (16, 16, 16)
    lattice = Matrix{Float64}(I, 3, 3)
    Δs = 0.05

    # Pre-compute frequently-used shifted ops + make_symmetric
    prep = Dict{Int, NamedTuple}()
    # Core groups + tricky groups from docs:
    # 227 Fd-3m: non-symmorphic d-glide, hit _is_pmmm_like bug in M2+Q
    # 230 Ia-3d: non-symmorphic screw axes, 25% M2+Q perf impact
    # 70  Fddd:  F-centering non-cubic, M7 breakthrough case
    # 72  Ibam:  low-symmetry I-centering (|G|=16)
    # 10  P2/m:  anisotropic L=(2,2,1), monoclinic edge case
    for sg in [221, 225, 229, 63, 47, 227, 230, 70, 72, 10]
        ops = get_ops(sg, 3, N16)
        _, ops_s = find_optimal_shift(ops, N16)
        spec = calc_spectral_asu(ops_s, 3, N16)
        u = make_symmetric(ops_s, N16)
        prep[sg] = (; ops, ops_s, spec, u)
    end

    # ── 4. Selective G0 vs FFTW (Pm-3m) ─────────────────────────────────
    @testset "Selective G0 vs FFTW (Pm-3m)" begin
        p = prep[221]
        ref = fft(ComplexF64.(p.u))
        ref_spec = [ref[(get_k_vector(p.spec, i) .+ 1)...] for i in 1:length(p.spec.points)]

        plan_sel = plan_krfft_selective(p.spec, p.ops_s)
        F_sel = execute_selective_krfft!(plan_sel, p.spec, p.u)

        err = maximum(abs.(F_sel .- ref_spec)) / maximum(abs.(ref_spec))
        @test err < 1e-12
    end

    # ── 5. G0 ASU vs FFTW (Fm-3m) ───────────────────────────────────────
    @testset "G0 ASU vs FFTW (Fm-3m)" begin
        p = prep[225]
        plan_asu = plan_krfft_g0asu(p.spec, p.ops_s)
        F_asu = execute_g0asu_krfft!(plan_asu, p.spec, p.u)

        F_ref = fft(complex(p.u))
        err = maximum(1:length(p.spec.points)) do i
            hv = get_k_vector(p.spec, i)
            ci = CartesianIndex(Tuple(mod.(hv, N16) .+ 1))
            abs(F_asu[i] - F_ref[ci])
        end
        @test err < 1e-10
    end

    # ── 6. G0 ASU Backward Roundtrip (Im-3m) ────────────────────────────
    @testset "G0 ASU Backward Roundtrip (Im-3m)" begin
        p = prep[229]
        fplan = plan_krfft_g0asu(p.spec, p.ops_s)
        F_asu = execute_g0asu_krfft!(fplan, p.spec, p.u)

        bplan = plan_krfft_g0asu_backward(p.spec, p.ops_s)
        u_out = zeros(N16...)
        execute_g0asu_ikrfft!(bplan, p.spec, copy(F_asu), u_out)

        @test maximum(abs.(p.u .- u_out)) < 1e-10
    end

    # ── 7-8. Centering Fold Plan + Fold vs FFT ───────────────────────────
    @testset "Centering Fold" begin
        @testset "Plan metadata" begin
            pf = plan_centering_fold(CentF, (16,16,16))
            @test pf.centering == CentF
            @test pf.n_channels == 2
            @test pf.H == (8,8,8)

            pi = plan_centering_fold(CentI, (16,16,16))
            @test pi.centering == CentI
            @test pi.n_channels == 4
        end

        @testset "Fold+FFT vs direct FFT (Im-3m)" begin
            p = prep[229]
            M = N16 .÷ 2
            f0 = Float64[p.u[2i-1,2j-1,2k-1] for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
            G0_ref = fft(f0)

            fp = plan_centering_fold(CentI, M)
            centering_fold!(fp, f0)
            fft_channels!(fp)
            G0 = zeros(ComplexF64, M)
            assemble_G0!(G0, fp)

            @test maximum(abs.(G0 .- G0_ref)) < 1e-10
        end
    end

    # ── 9. Centered KRFFT vs Full FFT (Fm-3m) ───────────────────────────
    @testset "Centered KRFFT vs FFT (Fm-3m)" begin
        p = prep[225]
        F_ref = fft(p.u)
        plan_c = plan_krfft_centered(p.spec, p.ops_s)
        @test plan_c isa CenteredKRFFTPlan

        execute_centered_krfft!(plan_c, p.u)
        spec_out = plan_c.krfft_plan.output_buffer

        max_err = 0.0
        for (i, _) in enumerate(p.spec.points)
            h = get_k_vector(p.spec, i)
            fref = F_ref[mod(h[1],N16[1])+1, mod(h[2],N16[2])+1, mod(h[3],N16[3])+1]
            max_err = max(max_err, abs(spec_out[i] - fref))
        end
        @test max_err < 1e-8
    end

    # ── 10-11. Q-Fused vs Full-Grid (Pmmm + Fm-3m) ──────────────────────
    @testset "Q-Fused KRFFT" begin
        @testset "Pmmm (SG 47)" begin
            p = prep[47]
            plan_q = plan_m2_q(N16, 47, 3, Δs, lattice)
            @test plan_q isa M2QPlan
            @test plan_q.L == (2, 2, 2)

            M = plan_q.M  # already NTuple
            f0 = zeros(Float64, M)
            fullgrid_to_subgrid!(f0, p.u, plan_q)

            f_ref_full = fullgrid_reference(p.u, N16, Δs, lattice, [2,2,2])

            f0_q = copy(f0)
            execute_m2_q!(plan_q, f0_q)

            @test maximum(abs.(f0_q .- f_ref_full)) < 1e-10
        end

        @testset "Fm-3m (SG 225)" begin
            p = prep[225]
            plan_q = plan_m2_q(N16, 225, 3, Δs, lattice)
            @test plan_q isa M2QPlan

            M = Tuple(plan_q.M)
            f0 = zeros(Float64, M)
            fullgrid_to_subgrid!(f0, p.u, plan_q)

            f_ref_full = fullgrid_reference(p.u, N16, Δs, lattice, plan_q.L)

            f0_q = copy(f0)
            execute_m2_q!(plan_q, f0_q)

            @test maximum(abs.(f0_q .- f_ref_full)) < 1e-10
        end
    end

    # ── 12. M7 Fold/Unfold Roundtrip (Im-3m) ────────────────────────────
    @testset "M7 Fold/Unfold Roundtrip (Im-3m)" begin
        p = prep[229]
        M = N16 .÷ 2
        f0 = Float64[p.u[2i-1,2j-1,2k-1] for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
        f0_backup = copy(f0)

        fp = plan_centering_fold(CentI, M)
        centering_fold!(fp, f0)
        centering_unfold!(fp, f0)

        @test maximum(abs.(f0 .- f0_backup)) < 1e-12
    end

    # ── 13. M7 vs M2+Q (Fm-3m) ──────────────────────────────────────────
    @testset "M7 vs M2+Q (Fm-3m)" begin
        p = prep[225]
        M = N16 .÷ 2
        f0_m7 = Float64[p.u[2i-1,2j-1,2k-1] for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
        f0_m2 = copy(f0_m7)

        m2q = plan_m2_q(N16, 225, 3, Δs, lattice)
        execute_m2_q!(m2q, f0_m2)

        m7 = plan_m7_scft(N16, 225, 3, Δs, lattice)
        execute_m7_scft!(m7, f0_m7)

        @test maximum(abs.(f0_m7 .- f0_m2)) < 1e-12
    end

    # ── 14. Centered Backward Roundtrip (Fm-3m) ─────────────────────────
    @testset "Centered Backward Roundtrip (Fm-3m)" begin
        p = prep[225]
        fwd = plan_krfft_centered(p.spec, p.ops_s)
        bwd = plan_centered_ikrfft(p.spec, p.ops_s, fwd)

        M = N16 .÷ 2
        f0_backup = Float64[p.u[2i-1,2j-1,2k-1] for i in 1:M[1], j in 1:M[2], k in 1:M[3]]

        pack_stride_real!(fwd.f0_buffer, p.u)
        F_spec = copy(CrystallographicFFT.KRFFT.fft_reconstruct_centered!(fwd))

        f0_out = zeros(Float64, M...)
        execute_centered_ikrfft!(bwd, F_spec, f0_out)

        @test maximum(abs.(f0_out .- f0_backup)) < 1e-10
    end

    # ── 15. Centered SCFT vs FFT (Im-3m) ────────────────────────────────
    @testset "Centered SCFT vs FFT (Im-3m)" begin
        p = prep[229]
        f0 = extract_subgrid(p.u, N16, [2,2,2])
        f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, [2,2,2])

        scft = plan_centered_scft(p.spec, p.ops_s, N16, Δs, lattice)
        execute_centered_scft!(scft, f0)

        @test maximum(abs.(f0 .- f0_ref)) < 1e-12
    end

    # ── 16-17. M2 SCFT vs FFT (Pm-3m + Fm-3m) ──────────────────────────
    @testset "M2 SCFT" begin
        @testset "Pm-3m vs FFT" begin
            p = prep[221]
            L = auto_L(p.ops_s)
            f0 = extract_subgrid(p.u, N16, L)
            f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, L)

            scft = plan_m2_scft(N16, 221, 3, Δs, lattice)
            execute_m2_scft!(scft, f0)

            @test maximum(abs.(f0 .- f0_ref)) < 1e-10
        end

        @testset "Fm-3m vs FFT" begin
            p = prep[225]
            L = auto_L(p.ops_s)
            f0 = extract_subgrid(p.u, N16, L)
            f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, L)

            scft = plan_m2_scft(N16, 225, 3, Δs, lattice)
            execute_m2_scft!(scft, f0)

            @test maximum(abs.(f0 .- f0_ref)) < 1e-10
        end
    end

    # ── 18. update_m2_kernel! (Pm-3m) ────────────────────────────────────
    @testset "update_m2_kernel!" begin
        p = prep[221]
        L = auto_L(p.ops_s)

        scft = plan_m2_scft(N16, 221, 3, 0.05, lattice)
        f0_ds05 = extract_subgrid(p.u, N16, L)
        execute_m2_scft!(scft, f0_ds05)

        update_m2_kernel!(scft, N16, 221, 3, 0.10, lattice)
        f0_ds10 = extract_subgrid(p.u, N16, L)
        execute_m2_scft!(scft, f0_ds10)

        scft2 = plan_m2_scft(N16, 221, 3, 0.10, lattice)
        f0_ref = extract_subgrid(p.u, N16, L)
        execute_m2_scft!(scft2, f0_ref)

        @test maximum(abs.(f0_ds10 .- f0_ref)) < 1e-14
        @test maximum(abs.(f0_ds05 .- f0_ds10)) > 1e-4
    end

    # ====================================================================
    # Tricky space groups (from docs analysis)
    # ====================================================================

    # ── 19. M2 SCFT: non-symmorphic groups (Fd-3m, Ia-3d) ───────────────
    # These groups triggered the _is_pmmm_like bug: d-glide/screw-axis
    # fractional translations were mis-routed to the separable butterfly.
    # See docs/implementation/m2_scft_implementation.md §M2+Q 精度修复
    @testset "M2 SCFT non-symmorphic" begin
        for (sg, name) in [(227, "Fd-3m"), (230, "Ia-3d")]
            @testset "$name (SG$sg)" begin
                p = prep[sg]
                L = auto_L(p.ops_s)
                f0 = extract_subgrid(p.u, N16, L)
                f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, L)

                scft = plan_m2_scft(N16, sg, 3, Δs, lattice)
                execute_m2_scft!(scft, f0)

                @test maximum(abs.(f0 .- f0_ref)) < 1e-10
            end
        end
    end

    # ── 20. M2 SCFT: anisotropic L (P2/m, L=[2,2,1]) ────────────────────
    # Monoclinic group with L=(2,2,1): only 4× FFT reduction, exercises
    # non-cubic stride logic. See docs/design/g0_asu_generalization_analysis.md
    @testset "M2 SCFT anisotropic L (P2/m)" begin
        p = prep[10]
        L = auto_L(p.ops_s)
        @test L != [2, 2, 2]  # confirm anisotropic
        f0 = extract_subgrid(p.u, N16, L)
        f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, L)

        scft = plan_m2_scft(N16, 10, 3, Δs, lattice)
        execute_m2_scft!(scft, f0)

        @test maximum(abs.(f0 .- f0_ref)) < 1e-10
    end

    # ── 21. M6 G0 ASU: Fd-3m (cubic, F-centering, non-symmorphic) ───────
    # Tests has_cubic_p3c_symmetry dispatch for F-centered cubic group.
    # See docs/implementation/general_g0_asu.md §7.1
    @testset "G0 ASU vs FFTW (Fd-3m)" begin
        p = prep[227]
        plan_asu = plan_krfft_g0asu(p.spec, p.ops_s)
        F_asu = execute_g0asu_krfft!(plan_asu, p.spec, p.u)

        F_ref = fft(complex(p.u))
        err = maximum(1:length(p.spec.points)) do i
            hv = get_k_vector(p.spec, i)
            ci = CartesianIndex(Tuple(mod.(hv, N16) .+ 1))
            abs(F_asu[i] - F_ref[ci])
        end
        @test err < 1e-10
    end

    # ── 22. M6 G0 ASU: Ia-3d (cubic, I-centering, screw axes) ───────────
    @testset "G0 ASU vs FFTW (Ia-3d)" begin
        p = prep[230]
        plan_asu = plan_krfft_g0asu(p.spec, p.ops_s)
        F_asu = execute_g0asu_krfft!(plan_asu, p.spec, p.u)

        F_ref = fft(complex(p.u))
        err = maximum(1:length(p.spec.points)) do i
            hv = get_k_vector(p.spec, i)
            ci = CartesianIndex(Tuple(mod.(hv, N16) .+ 1))
            abs(F_asu[i] - F_ref[ci])
        end
        @test err < 1e-10
    end

    # ── 23. M7 Centered KRFFT: Fddd (F-centering, non-cubic) ────────────
    # Breakthrough case: 32× FFT reduction on stride-2 subgrid.
    # See docs/design/fddd_centering_fold_verification.md
    @testset "Centered KRFFT vs FFT (Fddd)" begin
        p = prep[70]
        F_ref = fft(p.u)
        plan_c = plan_krfft_centered(p.spec, p.ops_s)
        @test plan_c isa CenteredKRFFTPlan

        execute_centered_krfft!(plan_c, p.u)
        spec_out = plan_c.krfft_plan.output_buffer

        max_err = 0.0
        for (i, _) in enumerate(p.spec.points)
            h = get_k_vector(p.spec, i)
            fref = F_ref[mod(h[1],N16[1])+1, mod(h[2],N16[2])+1, mod(h[3],N16[3])+1]
            max_err = max(max_err, abs(spec_out[i] - fref))
        end
        @test max_err < 1e-8
    end

    # ── 24. M7 Centered KRFFT: Ibam (I-centering, |G|=16) ───────────────
    # Low-symmetry I-centering: 4-channel WHT fold, marginal speedup range.
    # See docs/design/centering_fold_on_subgrid_plan.md §1.2
    @testset "Centered KRFFT vs FFT (Ibam)" begin
        p = prep[72]
        F_ref = fft(p.u)
        plan_c = plan_krfft_centered(p.spec, p.ops_s)
        @test plan_c isa CenteredKRFFTPlan

        execute_centered_krfft!(plan_c, p.u)
        spec_out = plan_c.krfft_plan.output_buffer

        max_err = 0.0
        for (i, _) in enumerate(p.spec.points)
            h = get_k_vector(p.spec, i)
            fref = F_ref[mod(h[1],N16[1])+1, mod(h[2],N16[2])+1, mod(h[3],N16[3])+1]
            max_err = max(max_err, abs(spec_out[i] - fref))
        end
        @test max_err < 1e-8
    end

    # ── 25. M7 Centered SCFT: Fddd (F-centering non-cubic SCFT) ─────────
    @testset "Centered SCFT vs FFT (Fddd)" begin
        p = prep[70]
        f0 = extract_subgrid(p.u, N16, [2,2,2])
        f0_ref = fullgrid_reference(p.u, N16, Δs, lattice, [2,2,2])

        scft = plan_centered_scft(p.spec, p.ops_s, N16, Δs, lattice)
        execute_centered_scft!(scft, f0)

        @test maximum(abs.(f0 .- f0_ref)) < 1e-12
    end

end
