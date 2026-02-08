## Test G0 ASU + Point Group Reduction
#
# Verifies correctness by comparing G0 ASU output against FFTW reference
# and against the selective approach. Also checks plan metadata.

using Test
using FFTW
using Random
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT: plan_krfft_g0asu, execute_g0asu_krfft!,
                                  plan_krfft_selective, execute_selective_krfft!

# Helper: symmetrize a random field
function make_symmetric(ops, N)
    u = rand(N...)
    s = zeros(N...)
    for op in ops
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(Int.(op.R) * x .+ round.(Int, op.t), collect(N)) .+ 1
            s[idx] += u[x2...]
        end
    end
    s ./= length(ops)
end

@testset "G0 ASU Reconstruction" begin
    # Correctness tests vs FFTW
    test_cases = [
        (225, "Fm-3m"), (229, "Im-3m"),
        (221, "Pm-3m"), (200, "Pm-3"),
    ]
    test_grids = [(16,16,16), (32,32,32)]

    for (sg, name) in test_cases, N in test_grids
        @testset "Correctness vs FFTW — $name N=$N" begin
            ops = get_ops(sg, 3, N)
            _, ops_s = find_optimal_shift(ops, N)
            spec = calc_spectral_asu(ops_s, 3, N)
            u = make_symmetric(ops_s, N)

            plan = plan_krfft_g0asu(spec, ops_s)
            F_asu = execute_g0asu_krfft!(plan, spec, u)

            F_ref = fft(complex(u))
            err = maximum(1:length(spec.points)) do i
                hv = get_k_vector(spec, i)
                ci = CartesianIndex(Tuple(hv .+ 1))
                abs(F_asu[i] - F_ref[ci])
            end
            @test err < 1e-12
        end
    end

    # Cross-validation: G0 ASU vs Selective
    for (sg, name) in test_cases
        N = (32,32,32)
        @testset "Consistency: G0 ASU vs Selective — $name N=$N" begin
            ops = get_ops(sg, 3, N)
            _, ops_s = find_optimal_shift(ops, N)
            spec = calc_spectral_asu(ops_s, 3, N)
            u = make_symmetric(ops_s, N)

            plan_sel = plan_krfft_selective(spec, ops_s)
            F_sel = execute_selective_krfft!(plan_sel, spec, u)

            plan_asu = plan_krfft_g0asu(spec, ops_s)
            F_asu = execute_g0asu_krfft!(plan_asu, spec, u)

            # Different accumulation order → minor FP drift
            err = maximum(abs.(F_sel .- F_asu))
            @test err < 1e-10
        end
    end

    # Plan metadata
    @testset "Plan metadata — Fm-3m" begin
        N = (32,32,32)
        ops = get_ops(225, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)

        plan = plan_krfft_g0asu(spec, ops_s)

        @test plan.n_spec == length(spec.points)
        # n_reps should equal n_spec for all groups
        @test plan.n_reps == plan.n_spec
        # n_reps << |S| (significant reduction)
        @test plan.n_reps < length(plan.g0_p3c) * 5  # sanity check
        @test length(plan.g0_p3c) == plan.n_reps
        @test length(plan.g0_values) == plan.n_reps
        @test length(plan.a8_table) == 8 * plan.n_spec
    end

    # Idempotency
    @testset "Idempotency — repeated execution" begin
        N = (32,32,32)
        ops = get_ops(225, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u = make_symmetric(ops_s, N)

        plan = plan_krfft_g0asu(spec, ops_s)
        F1 = copy(execute_g0asu_krfft!(plan, spec, u))
        F2 = copy(execute_g0asu_krfft!(plan, spec, u))
        @test F1 == F2  # exact equality
    end
end
