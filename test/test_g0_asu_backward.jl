## Test G0 ASU Backward Transform (P3c-level)
#
# Verifies forward→backward roundtrip consistency.

using Test
using FFTW
using Random
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT: plan_krfft_g0asu, execute_g0asu_krfft!,
                                  plan_krfft_g0asu_backward, execute_g0asu_ikrfft!

# Helper: symmetrize a random field
function make_symmetric(ops, N)
    Random.seed!(42)
    u = rand(N...)
    s = zeros(N...)
    for op in ops
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(round.(Int, op.R) * x .+ round.(Int, op.t), collect(N)) .+ 1
            s[idx] += u[x2...]
        end
    end
    s ./= length(ops)
end

@testset "G0 ASU Backward Transform (IKRFFT)" begin

    # Only test groups where the forward G0 ASU KRFFT is verified correct.
    # P4₂/mnm (136) and Fddd (70) have pre-existing forward issues.
    test_cases = [
        (225, "Fm-3m"),
        (229, "Im-3m"),
        (221, "Pm-3m"),
        (200, "Pm-3"),
    ]

    @testset "Forward-backward roundtrip — $name N=$N" for
            (sg, name) in test_cases, N in [(16,16,16)]

        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u = make_symmetric(ops_s, N)

        # Forward: u → F(h)
        fplan = plan_krfft_g0asu(spec, ops_s)
        F_asu = execute_g0asu_krfft!(fplan, spec, u)

        # Verify forward correctness vs FFTW
        F_ref = fft(complex(u))
        fwd_err = maximum(1:length(spec.points)) do i
            hv = get_k_vector(spec, i)
            ci = CartesianIndex(Tuple(mod.(hv, collect(N)) .+ 1))
            abs(F_asu[i] - F_ref[ci])
        end
        @test fwd_err < 1e-10

        # Backward: F(h) → u'
        bplan = plan_krfft_g0asu_backward(spec, ops_s)
        u_out = zeros(N...)
        execute_g0asu_ikrfft!(bplan, spec, copy(F_asu), u_out)

        max_err = maximum(abs.(u .- u_out))
        println("  $name N=$N: max roundtrip error = $(max_err)")
        @test max_err < 1e-10
    end

end
