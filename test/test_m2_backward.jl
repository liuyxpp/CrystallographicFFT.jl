using Pkg
Pkg.activate(".")
using Test
using LinearAlgebra
using FFTW
using CrystallographicFFT
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT

"""
    make_symmetric(ops, N)

Generate a truly symmetric real-valued field on the full grid N,
by averaging a random field over all symmetry operations.
"""
function make_symmetric(ops, N)
    u = randn(Tuple(N)...)
    u_sym = zeros(Tuple(N)...)
    dim = length(N)
    n_ops = length(ops)
    for op in ops
        R = round.(Int, op.R)
        t = round.(Int, op.t)
        for x in CartesianIndices(Tuple(N))
            xv = [x[dd] - 1 for dd in 1:dim]
            gx = Tuple(mod.(R * xv .+ t, collect(N)) .+ 1)
            u_sym[x] += u[gx...]
        end
    end
    u_sym ./= n_ops
    return u_sym
end

"""
    extract_subgrid(u_sym, L, M)

Extract the stride-L subgrid from the symmetric full-grid field.
Returns a flat real vector of length prod(M).
"""
function extract_subgrid(u_sym, L, M)
    dim = length(L)
    f0 = zeros(Float64, prod(M))
    for m in CartesianIndices(Tuple(M))
        mv = [m[dd] - 1 for dd in 1:dim]
        x = Tuple(L .* mv .+ 1)  # 1-based index into u_sym
        lin = 1
        stride = 1
        for dd in 1:dim
            lin += mv[dd] * stride
            stride *= M[dd]
        end
        f0[lin] = u_sym[x...]
    end
    return f0
end

"""
Generate a random subgrid input, run forward → backward, check roundtrip.

For groups where d = prod(L) < |G|, the forward KRFFT map is rank-deficient
for arbitrary subgrid data. In that case, only symmetric f₀ (extracted from
a symmetric full-grid field) produces a valid roundtrip. The `use_symmetric`
flag controls this:
  - `true`: generate symmetric full field, extract subgrid (always works)
  - `false`: random subgrid data (only works when d == |G|)
"""
function test_roundtrip(sg_num::Int, N_val::Int; dim::Int=3, atol::Float64=1e-10,
                        use_symmetric::Bool=true)
    N = ntuple(_ -> N_val, dim)

    # 1. Setup: get ops and shift
    direct_ops = get_ops(sg_num, dim, N)
    _, ops_shifted = find_optimal_shift(direct_ops, N)

    # 2. Compute spectral ASU
    spec = calc_spectral_asu(ops_shifted, dim, N)
    n_spec = length(spec.points)

    # 3. Plan forward and backward
    plan_fwd = plan_krfft(spec, ops_shifted)
    plan_bwd = plan_m2_backward(spec, ops_shifted)

    M = plan_fwd.subgrid_dims
    L = plan_fwd.L_factors[1]
    M_vol = prod(M)
    d = prod(L)

    # 4. Generate subgrid input
    if use_symmetric
        u_sym = make_symmetric(ops_shifted, N)
        f0 = extract_subgrid(u_sym, L, M)
    else
        f0 = randn(Float64, M_vol)
    end

    # 5. Forward transform
    plan_fwd.input_buffer .= complex.(f0)
    F_spec = copy(fft_reconstruct!(plan_fwd))

    # 6. Backward transform
    f0_recovered = execute_m2_backward!(plan_bwd, F_spec)

    # 7. Compare
    max_err = maximum(abs, real.(f0_recovered) .- f0)
    rel_err = max_err / max(maximum(abs, f0), 1e-15)
    return max_err, rel_err, n_spec, M, d, length(ops_shifted)
end

# ============================================================================
# Test Suite
# ============================================================================

@testset "M2 Backward Transform" begin

    # P-centering groups (M2 backward is the primary/only option)
    p_groups = [
        (47,  "Pmmm",    16),
        (47,  "Pmmm",    32),
        (123, "P4/mmm",  16),
        (123, "P4/mmm",  32),
        (136, "P42/mnm", 16),
        (136, "P42/mnm", 32),
        (221, "Pm-3m",   16),
        (221, "Pm-3m",   32),
        (10,  "P2/m",    16),
        (10,  "P2/m",    32),
        (2,   "P-1",     16),
        (2,   "P-1",     32),
    ]

    @testset "P-centering: $name N=$N_val" for (sg, name, N_val) in p_groups
        max_err, rel_err, n_spec, M, d, nG = test_roundtrip(sg, N_val)
        println("  $name (SG $sg) N=$N_val: M=$M n_spec=$n_spec d=$d |G|=$nG max_err=$max_err")
        @test max_err < 1e-10
    end

    # Centered groups with glide (M7 backward struggles, M2 backward is key alternative)
    centered_groups = [
        (230, "Ia-3d",   16),
        (230, "Ia-3d",   32),
        (70,  "Fddd",    16),
        (70,  "Fddd",    32),
        (69,  "Fmmm",    16),
        (69,  "Fmmm",    32),
        (139, "I4/mmm",  16),
        (139, "I4/mmm",  32),
        (227, "Fd-3m",   16),
        (227, "Fd-3m",   32),
    ]

    @testset "Centered: $name N=$N_val" for (sg, name, N_val) in centered_groups
        max_err, rel_err, n_spec, M, d, nG = test_roundtrip(sg, N_val)
        println("  $name (SG $sg) N=$N_val: M=$M n_spec=$n_spec d=$d |G|=$nG max_err=$max_err")
        @test max_err < 1e-10
    end

end
