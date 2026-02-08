using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT: plan_krfft_selective, execute_selective_krfft!,
    plan_krfft_sparse, execute_sparse_krfft!
using FFTW
using Random

"""
    make_symmetric(ops, N; seed=42)

Create a deterministic symmetric real-space field by projecting random noise
onto the symmetry manifold defined by `ops`.
"""
function make_symmetric(ops, N; seed=42)
    Random.seed!(seed)
    u = rand(Float64, N...)
    s = zeros(Float64, N...)
    for op in ops
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(Int.(op.R) * x .+ round.(Int, op.t), collect(N)) .+ 1
            s[idx] += u[x2...]
        end
    end
    s ./= length(ops)
    return s
end

@testset "Selective G0 Cascade" begin

    @testset "Correctness vs FFTW — $nm N=$N" for (sg, nm, N) in [
        (225, "Fm-3m", (16,16,16)),
        (225, "Fm-3m", (32,32,32)),
        (229, "Im-3m", (16,16,16)),
        (229, "Im-3m", (32,32,32)),
        (221, "Pm-3m", (16,16,16)),
        (221, "Pm-3m", (32,32,32)),
        (200, "Pm-3",  (16,16,16)),
        (200, "Pm-3",  (32,32,32)),
    ]
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u = make_symmetric(ops_s, N)

        # Reference: full FFT
        ref = fft(ComplexF64.(u))
        ref_spec = [ref[(get_k_vector(spec, i) .+ 1)...] for i in 1:length(spec.points)]

        # Selective G0
        plan_sel = plan_krfft_selective(spec, ops_s)
        F_sel = execute_selective_krfft!(plan_sel, spec, u)

        err = maximum(abs.(F_sel .- ref_spec)) / maximum(abs.(ref_spec))
        @test err < 1e-12
    end

    @testset "Consistency: Selective vs Sparse — $nm N=$N" for (sg, nm, N) in [
        (225, "Fm-3m", (32,32,32)),
        (229, "Im-3m", (32,32,32)),
        (221, "Pm-3m", (32,32,32)),
        (200, "Pm-3",  (32,32,32)),
    ]
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u = make_symmetric(ops_s, N)

        plan_sp = plan_krfft_sparse(spec, ops_s)
        F_sp = execute_sparse_krfft!(plan_sp, spec, u)

        plan_sel = plan_krfft_selective(spec, ops_s)
        F_sel = execute_selective_krfft!(plan_sel, spec, u)

        # Both should produce the same result (different accumulation order → ~1e-12 FP drift)
        err = maximum(abs.(F_sp .- F_sel))
        @test err < 1e-10
    end

    @testset "Plan metadata — Fm-3m" begin
        N = (32,32,32)
        ops = get_ops(225, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)

        plan = plan_krfft_selective(spec, ops_s)

        # G0 positions should be << M³
        M3 = prod(N .÷ 2)
        n_g0 = length(plan.g0_values)
        @test n_g0 < M3              # selective: not all G0 computed
        @test n_g0 > 0               # at least some computed
        @test n_g0 < 0.30 * M3       # Fm-3m: expect ~23.6%

        # A8 table should have exactly 8 entries per spectral point
        @test length(plan.a8_table) == 8 * plan.n_spec

        # Buffers should be (N/4)³
        M2 = Tuple(N .÷ 4)
        @test size(plan.buffers[1]) == M2
        @test length(plan.fft_flat) == 4 * prod(M2)
    end

    @testset "Idempotency — repeated execution" begin
        N = (16,16,16)
        ops = get_ops(225, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u = make_symmetric(ops_s, N)

        plan = plan_krfft_selective(spec, ops_s)
        F1 = copy(execute_selective_krfft!(plan, spec, u))
        F2 = copy(execute_selective_krfft!(plan, spec, u))

        @test F1 ≈ F2 atol=1e-15
    end
end
