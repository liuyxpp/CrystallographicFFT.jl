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

"""Symmetrize input so u(R_g x + t_g) = u(x) for all g."""
function make_symmetric(ops, N)
    u = rand(N...)
    s = zeros(N...)
    for op in ops, idx in CartesianIndices(u)
        x = collect(Tuple(idx)) .- 1
        x2 = mod.(round.(Int, op.R * x + op.t), collect(N)) .+ 1
        s[idx] += u[x2...]
    end
    s ./= length(ops)
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
        # These groups have only diagonal-R ops → parity split but NO orbit equivalence
        for (sg, name, N) in [
            (2,  "P-1",  (8,8,8)),
            (47, "Pmmm", (8,8,8)),
            (70, "Fddd", (8,8,8)),
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

    @testset "Orbit equivalence (mixing ops)" begin
        for (sg, name, N, G_order) in [
            (200, "Pm-3",    (8,8,8),  24),
            (136, "P42/mnm", (8,8,8),  16),
            (221, "Pm-3m",   (8,8,8),  48),
            (225, "Fm-3m",   (8,8,8), 192),
            (229, "Im-3m",   (8,8,8),  96),
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

    @testset "Multi-level recursion" begin
        # High-symmetry groups need larger N to exercise deep recursion
        for (sg, name, N, G_order) in [
            (225, "Fm-3m", (16,16,16), 192),
            (221, "Pm-3m", (16,16,16),  48),
            (200, "Pm-3",  (16,16,16),  24),
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

    @testset "Orbit equivalence formula verification" begin
        # Verify Y_equiv(h) = phase × Y_rep(R_formula h) at multiple tree levels
        N = (16, 16, 16)
        ops = get_ops(225, 3, N)  # Fm-3m
        u = make_symmetric(ops, N)
        root = build_recursive_tree(N, ops)

        @testset "Root level (P3c orbit)" begin
            # Root sector (1,0,0) is rep, sector (0,0,1) equiv via P3c
            # R_α * (1,0,0) = (0,0,1), so R_α maps rep→equiv
            Y_100 = extract_sector_fft(u, collect(N), [2,2,2], [1,0,0], [8,8,8])
            Y_001 = extract_sector_fft(u, collect(N), [2,2,2], [0,0,1], [8,8,8])
            M = (8, 8, 8)

            # Test: Y_001(h) = Y_100(R_α^{-1} h) ← because R_α maps rep→equiv
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

        @testset "Depth-1 sub-sector orbit" begin
            # Child 2 (p=100) splits dims 2,3 → sub-sectors
            # Rep p=(0,1,0) off=(1,2,0), equiv p=(0,0,1) off=(1,0,2)
            child2 = root.children[2]
            R_rel = round.(Int, child2.sector_rot[3])  # sector 3's relating op
            t_rel = child2.sector_trans[3]

            gc_rep = child2.children[2]  # rep child for sector 2
            Y_rep = extract_sector_fft(u, collect(N), gc_rep.scale, gc_rep.offset, gc_rep.subgrid_N)

            eq_offset = child2.scale .* [0,0,1] .+ child2.offset  # equiv sector offset
            Y_eq = extract_sector_fft(u, collect(N), gc_rep.scale, eq_offset, gc_rep.subgrid_N)
            M = Tuple(gc_rep.subgrid_N)

            # Compute offset correction δ = S^{-1}(R^{-1}(p_eq - t) - p_rep)
            S = gc_rep.scale
            p_eq_full = eq_offset
            p_rep_full = gc_rep.offset
            R_inv = round.(Int, inv(Float64.(R_rel)))
            delta = (R_inv * (p_eq_full .- round.(Int, t_rel)) .- p_rep_full) .÷ S

            # Y_eq(h) = exp(2πi h·R·δ/M) × Y_rep(R^T h)
            # But also test Y_eq(h) = Y_rep(R h) (the simpler formula that worked)
            for (label, use_Rt, use_delta) in [
                ("R^T, no δ", true, false),
                ("R^T, with δ", true, true),
                ("R, no δ", false, false),
            ]
                max_err = 0.0
                for ci in CartesianIndices(M)
                    h = collect(Tuple(ci)) .- 1
                    if use_Rt
                        h_rot = [sum(R_rel[j,d]*h[j] for j in 1:3) for d in 1:3]
                    else
                        h_rot = [sum(R_rel[d,j]*h[j] for j in 1:3) for d in 1:3]
                    end
                    idx = Tuple(mod.(h_rot, collect(M)) .+ 1)
                    val = Y_rep[idx...]
                    if use_delta
                        Rdelta = R_rel * delta
                        phase = sum(h[d] * Rdelta[d] / M[d] for d in 1:3)
                        val *= cispi(2 * phase)
                    end
                    err = abs(Y_eq[ci] - val)
                    max_err = max(max_err, err)
                end
                println("  Sub-level: $label → err=$(round(max_err, sigdigits=3))")
                if label == "R, no δ"
                    @test max_err < 1e-12  # Currently this is what works
                end
            end
        end
    end

    @testset "Child buffer correctness" begin
        # Verify each root child's FFT buffer matches manual sector FFT
        N = (16, 16, 16)
        ops = get_ops(225, 3, N)
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
