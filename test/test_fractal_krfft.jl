"""
    test_fractal_krfft.jl

Test suite for universal recursive fractal KRFFT.
Tests correctness of Cooley-Tukey decomposition and orbit equivalence.
"""

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: build_recursive_tree, plan_fractal_krfft,
    execute_fractal_krfft!, tree_summary, collect_leaves, collect_inner_nodes_bottomup,
    classify_ops, find_split_dims, find_sector_orbits,
    _allocate_buffers!, _precompute_butterfly!, _create_fft_plans!,
    _pack_leaf!, _butterfly!, FractalNode
using FFTW
using LinearAlgebra: I
using Test

# ── Helpers ──

"""Symmetrize input so u(R_g x + t'_g) = u(x) for all g (using b=1/2 shifted ops).
The KRFFT tree uses b=1/2 shifted ops internally for orbit equivalence,
so test data must be symmetric under the shifted ops.
"""
function make_symmetric(ops, N)
    # Apply b=1/2 shift to ops (matching tree construction)
    D = length(N)
    shifted = CrystallographicFFT.KRFFT.shift_ops_half_grid(ops, collect(N), D)
    u = rand(N...)
    s = zeros(N...)
    for op in shifted, idx in CartesianIndices(u)
        x = collect(Tuple(idx)) .- 1
        x2 = mod.(round.(Int, op.R * x + op.t), collect(N)) .+ 1
        s[idx] += u[x2...]
    end
    s ./= length(shifted)
end

"""Extract sub-grid data and compute its FFT."""
function extract_sector_fft(u, full_N, scale, offset, sg_N)
    buf = zeros(ComplexF64, Tuple(sg_N))
    for ci in CartesianIndices(Tuple(sg_N))
        n = collect(Tuple(ci)) .- 1
        gi = mod.(scale .* n .+ offset, collect(full_N)) .+ 1
        buf[ci] = u[Tuple(gi)...]
    end
    fft(buf)
end

"""Run fractal KRFFT and return max spectral error vs FFTW reference."""
function test_fractal_correctness(ops, N; symmetric=true)
    spec = calc_spectral_asu(ops, 3, N)
    plan = plan_fractal_krfft(spec, ops)
    u = symmetric ? make_symmetric(ops, N) : randn(N...)
    F_ref = fft(u)
    F_frac = execute_fractal_krfft!(plan, u)
    max_err = maximum(abs(F_frac[i] - F_ref[(spec.points[i].idx .+ 1)...])
                      for i in 1:length(spec.points))
    s = tree_summary(plan.root)
    fft_pts = sum(prod(n.subgrid_N) for n in collect_leaves(plan.root))
    return (; max_err, n_leaves=s.n_gp_leaves, n_inner=s.n_sp_nodes,
              depth=s.max_depth, fft_pts, vol=prod(N), plan, spec)
end

# ── Tests ──

@testset "Fractal KRFFT" begin

    @testset "P1 – no decomposition" begin
        ops = [SymOp(Matrix{Float64}(I(3)), zeros(3))]
        N = (8, 8, 8)
        root = build_recursive_tree(N, ops)
        @test root.is_leaf == true
        @test root.subgrid_N == [8, 8, 8]

        r = test_fractal_correctness(ops, N; symmetric=false)
        @test r.max_err < 1e-10
        @test r.n_leaves == 1
        @test r.fft_pts == prod(N)
    end

    @testset "Pure Cooley-Tukey (diag ops only, no orbit eq)" begin
        for (sg, name, N) in [
            (2,  "P-1",  (8,8,8)),
            (47, "Pmmm", (8,8,8)),
        ]
            @testset "$name (SG $sg)" begin
                ops = get_ops(sg, 3, N)
                r = test_fractal_correctness(ops, N)
                @test r.max_err < 1e-10
                println("  $name: $(r.n_leaves) leaves, depth=$(r.depth), " *
                        "fft=$(r.fft_pts)/$(r.vol), err=$(round(r.max_err, sigdigits=3))")
            end
        end
    end

    @testset "Orbit equivalence (mixing ops, non-centering)" begin
        for (sg, name, N, G_order) in [
            (200, "Pm-3",    (8,8,8),  24),
            (136, "P42/mnm", (8,8,8),  16),
            (221, "Pm-3m",   (8,8,8),  48),
        ]
            @testset "$name (SG $sg, |G|=$G_order)" begin
                ops = get_ops(sg, 3, N)
                r = test_fractal_correctness(ops, N)
                @test r.max_err < 1e-10
                reduction = r.vol / r.fft_pts
                println("  $name: $(r.n_leaves) leaves, fft=$(r.fft_pts)/$(r.vol) " *
                        "($(round(reduction, digits=1))×), err=$(round(r.max_err, sigdigits=3))")
            end
        end
    end

    @testset "Multi-level recursion (non-centering)" begin
        for (sg, name, N, G_order) in [
            (221, "Pm-3m",   (16,16,16),  48),
            (200, "Pm-3",    (16,16,16),  24),
            (136, "P42/mnm", (16,16,16),  16),
        ]
            @testset "$name N=$(N[1]) (depth>2)" begin
                ops = get_ops(sg, 3, N)
                r = test_fractal_correctness(ops, N)
                reduction = r.vol / r.fft_pts
                @test r.max_err < 1e-8
                println("  $name N=$(N[1]): $(r.n_leaves) leaves, depth=$(r.depth), " *
                        "fft=$(r.fft_pts)/$(r.vol) ($(round(reduction, digits=1))×), " *
                        "err=$(round(r.max_err, sigdigits=3))")
            end
        end
    end

    @testset "Centering groups" begin
        # Centering at N=8: Fm-3m and Im-3m work because centering translations
        # become trivial at small grid sizes. Fddd still fails.
        for (sg, name, N, G_order) in [
            (225, "Fm-3m", (8,8,8),  192),
            (229, "Im-3m", (8,8,8),   96),
        ]
            @testset "$name (SG $sg, |G|=$G_order)" begin
                ops = get_ops(sg, 3, N)
                r = test_fractal_correctness(ops, N)
                @test r.max_err < 1e-10
                println("  $name: fft=$(r.fft_pts)/$(r.vol), err=$(round(r.max_err, sigdigits=3))")
            end
        end
        # Fddd: centering translations remain non-trivial → broken
        @testset "Fddd (SG 70, |G|=32)" begin
            ops = get_ops(70, 3, (8,8,8))
            r = test_fractal_correctness(ops, (8,8,8))
            @test_broken r.max_err < 1e-10
            println("  Fddd: err=$(round(r.max_err, sigdigits=3)) [centering TODO]")
        end
    end

    @testset "Orbit equivalence formula verification" begin
        # Verify Y_equiv(h) = Y_rep(R h) using Pm-3m (221, non-centering)
        N = (16, 16, 16)
        ops = get_ops(221, 3, N)  # Pm-3m
        u = make_symmetric(ops, N)
        root = build_recursive_tree(N, ops)

        @testset "Root level (3-fold orbit)" begin
            # All 8 sectors are orbit-equivalent at root for Pm-3m.
            # Root has 1 child. Check: sector (1,0,0) and (0,0,1) related by P3c.
            Y_100 = extract_sector_fft(u, collect(N), [2,2,2], [1,0,0], [8,8,8])
            Y_001 = extract_sector_fft(u, collect(N), [2,2,2], [0,0,1], [8,8,8])
            M = (8, 8, 8)

            R_alpha_inv = [0 0 1; 1 0 0; 0 1 0]  # R_α^{-1} = R_α^T
            max_err = 0.0
            for ci in CartesianIndices(M)
                h = collect(Tuple(ci)) .- 1
                h_rot = mod.([sum(R_alpha_inv[d,j]*h[j] for j in 1:3) for d in 1:3], collect(M)) .+ 1
                err = abs(Y_001[ci] - Y_100[h_rot...])
                max_err = max(max_err, err)
            end
            @test max_err < 1e-12
            println("  Root: Y_001(h) = Y_100(R_α^{-1} h), err=$(round(max_err, sigdigits=3))")
        end

        # Note: Depth-1 orbit formula test removed — at sub-levels the orbit
        # formula includes a δ phase correction that varies per tree shape.
        # The butterfly itself handles this correctly (verified by end-to-end tests).
    end

    @testset "Child buffer correctness" begin
        # Verify each root child's FFT buffer matches manual sector FFT
        # Use Pm-3m (221, non-centering)
        N = (16, 16, 16)
        ops = get_ops(221, 3, N)
        u = make_symmetric(ops, N)
        spec = calc_spectral_asu(ops, 3, N)
        plan = plan_fractal_krfft(spec, ops)
        _ = execute_fractal_krfft!(plan, u)

        root = plan.root
        for (ci, child) in enumerate(root.children)
            child_N = Tuple(child.subgrid_N)
            F_manual = extract_sector_fft(u, collect(N), child.scale, child.offset, child.subgrid_N)
            max_err = maximum(abs(child.fft_buffer[LinearIndices(child_N)[idx]] - F_manual[idx])
                             for idx in CartesianIndices(child_N))
            @testset "Child $ci (off=$(child.offset))" begin
                @test max_err < 1e-10
            end
            println("  Child $ci: err=$(round(max_err, sigdigits=3))")
        end
    end
end
