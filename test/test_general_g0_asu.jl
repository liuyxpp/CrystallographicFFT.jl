## Test General G0 ASU — All Crystal Systems
#
# Verifies correctness of the generalized G0 ASU (all space groups)
# by comparing against FFTW reference and the existing cubic G0 ASU.

using Test
using FFTW
using Random
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT: plan_krfft_g0asu, execute_g0asu_krfft!,
                                  plan_krfft_g0asu_general, execute_general_g0asu_krfft!,
                                  GeneralG0ASUPlan, auto_L

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

# Helper: compare against FFTW reference with proper mod wrapping
function max_error_vs_fft(F_asu, spec, F_ref, N)
    maximum(1:length(spec.points)) do i
        hv = get_k_vector(spec, i)
        ci = CartesianIndex(Tuple(mod.(hv, collect(N)) .+ 1))
        abs(F_asu[i] - F_ref[ci])
    end
end

@testset "General G0 ASU — All Crystal Systems" begin
    # Groups where auto_L should give L > [1,1,1]
    test_cases = [
        # (space_group_number, name, crystal_system)
        (2,   "P-1",        "triclinic"),
        (10,  "P2/m",       "monoclinic"),
        (47,  "Pmmm",       "orthorhombic"),
        (123, "P4/mmm",     "tetragonal"),
    ]

    # Cubic groups: test both dispatch paths
    cubic_cases = [
        (200, "Pm-3",  "cubic"),
        (221, "Pm-3m", "cubic"),
    ]

    test_grids = [(16,16,16), (32,32,32)]

    @testset "Correctness vs FFTW — $name ($system) N=$N" for (sg, name, system) in test_cases, N in test_grids
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)

        L = auto_L(ops_s)
        n_sub = prod(L)
        if n_sub <= 1
            @info "  Skipping $name: auto_L=$L, no reduction"
            continue
        end

        u = make_symmetric(ops_s, N)
        plan = plan_krfft_g0asu_general(spec, ops_s)
        @test plan isa GeneralG0ASUPlan
        F_asu = execute_general_g0asu_krfft!(plan, spec, u)

        F_ref = fft(complex(u))
        err = max_error_vs_fft(F_asu, spec, F_ref, N)
        @test err < 1e-9
        @info "  $name N=$N: L=$L, n_ops=$(plan.n_ops), n_spec=$(plan.n_spec), max_err=$(round(err, sigdigits=3))"
    end

    @testset "Auto-dispatch — $name (cubic) N=$N" for (sg, name, _) in cubic_cases, N in test_grids
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u = make_symmetric(ops_s, N)

        # Dispatch should route cubic to existing G0ASUPlan
        plan_dispatched = plan_krfft_g0asu(spec, ops_s)
        F_dispatched = execute_g0asu_krfft!(plan_dispatched, spec, u)

        # Also test general plan directly for cubic groups
        plan_general = plan_krfft_g0asu_general(spec, ops_s)
        @test plan_general isa GeneralG0ASUPlan
        F_general = execute_general_g0asu_krfft!(plan_general, spec, u)

        # Both should match FFTW
        F_ref = fft(complex(u))
        err_dispatched = max_error_vs_fft(F_dispatched, spec, F_ref, N)
        err_general = max_error_vs_fft(F_general, spec, F_ref, N)
        @test err_dispatched < 1e-9
        @test err_general < 1e-9

        # Cross-check: general ≈ specialized
        cross_err = maximum(abs.(F_dispatched .- F_general))
        @test cross_err < 1e-9
        @info "  $name N=$N: dispatch_err=$(round(err_dispatched, sigdigits=3)), general_err=$(round(err_general, sigdigits=3)), cross=$(round(cross_err, sigdigits=3))"
    end

    @testset "Plan metadata" begin
        N = (32,32,32)
        ops = get_ops(47, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        plan = plan_krfft_g0asu_general(spec, ops_s)

        @test plan.n_spec == length(spec.points)
        @test plan.L == [2,2,2]
        @test plan.subgrid_dims == [16,16,16]
        @test plan.n_ops <= 8  # n_parities, not total group order
        @test length(plan.g0_weights) == plan.n_spec
    end

    @testset "auto_L coverage" begin
        for (sg, name, _) in test_cases
            N = (32,32,32)
            ops = get_ops(sg, 3, N)
            _, ops_s = find_optimal_shift(ops, N)
            L = auto_L(ops_s)
            n_sub = prod(L)
            @test n_sub >= 2
            @info "  $name: auto_L=$L ($(n_sub)× reduction)"
        end
    end
end
