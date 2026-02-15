using Test
using FFTW
using LinearAlgebra
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.QFusedKRFFT

"""
    make_symmetric(f, ops, N)

Symmetrize a real field f under the given space group operations.
Returns Σ_g f(g⁻¹ x) / |G|
"""
function make_symmetric(f::Array{Float64}, ops::Vector{<:SymOp}, N::Tuple)
    D = length(N)
    f_sym = zeros(Float64, N)
    x_rot = zeros(Int, D)

    for ci in CartesianIndices(f)
        x = [ci[d] - 1 for d in 1:D]  # 0-based
        for op in ops
            # Apply op: x' = R·x + t (mod N)
            for d1 in 1:D
                s = round(Int, op.t[d1])
                for d2 in 1:D
                    s += op.R[d1, d2] * x[d2]
                end
                x_rot[d1] = mod(s, N[d1])
            end
            f_sym[x_rot[1]+1, x_rot[2]+1, x_rot[3]+1] += f[ci]
        end
    end
    f_sym ./= length(ops)
    return f_sym
end

"""
    reference_diffusion_step(f_sym, Δs, lattice, N)

Full-grid FFT → K multiply → IFFT reference implementation.
"""
function reference_diffusion_step(f_sym::Array{Float64}, Δs::Float64,
                                  lattice::AbstractMatrix, N::Tuple)
    D = length(N)
    recip_B = 2π * inv(lattice)'

    F = fft(complex(f_sym))

    # Apply diffusion kernel K(h) = exp(-|K(h)|² Δs)
    for ci in CartesianIndices(F)
        h_vec = [ci[d] - 1 for d in 1:D]
        # Wrap to centered frequencies
        for d in 1:D
            if h_vec[d] >= N[d] ÷ 2
                h_vec[d] -= N[d]
            end
        end
        K_phys = recip_B * h_vec
        k2 = dot(K_phys, K_phys)
        F[ci] *= exp(-k2 * Δs)
    end

    f_new = real.(ifft(F))
    return f_new
end

@testset "Q-Fused KRFFT" begin
    # Common parameters
    Δs = 0.05

    @testset "Pmmm (SG 47) N=$N" for N in [(16,16,16), (32,32,32)]
        sg_num = 47
        dim = 3
        lattice = Matrix{Float64}(I, 3, 3)

        # Build plan
        plan = plan_m2_q(N, sg_num, dim, Δs, lattice)

        @test plan isa M2QPlan
        @test plan.L == [2, 2, 2]
        @test prod(plan.L) == 8
        @test all(plan.M .== [N[d] ÷ 2 for d in 1:3])

        # Generate symmetric test field
        ops = get_ops(sg_num, dim, N)
        _, shifted_ops = find_optimal_shift(ops, N)
        f_random = randn(N)
        f_sym = make_symmetric(f_random, shifted_ops, N)

        # Extract subgrid from full symmetric field
        M = Tuple(plan.M)
        L = plan.L
        f0 = zeros(Float64, M)
        fullgrid_to_subgrid!(f0, f_sym, plan)

        # Reference: full-grid FFT → K → IFFT on shifted field
        f_ref = reference_diffusion_step(f_sym, Δs, lattice, N)

        # Extract reference subgrid
        f0_ref = zeros(Float64, M)
        fullgrid_to_subgrid!(f0_ref, f_ref, plan)

        # Q-fused path
        f0_q = copy(f0)
        execute_m2_q!(plan, f0_q)

        # Compare subgrid results
        err = maximum(abs.(f0_q .- f0_ref))
        @test err < 1e-10

        # Grid conversion round-trip test
        f_expanded = zeros(Float64, N)
        subgrid_to_fullgrid!(f_expanded, f0, plan)
        f0_roundtrip = zeros(Float64, M)
        fullgrid_to_subgrid!(f0_roundtrip, f_expanded, plan)
        @test f0_roundtrip ≈ f0 atol=1e-14
    end

    @testset "P4/mmm (SG 123) N=$N" for N in [(16,16,16), (32,32,32)]
        sg_num = 123
        dim = 3
        # Tetragonal: a=b≠c
        lattice = [1.0 0 0; 0 1.0 0; 0 0 1.5]

        plan = plan_m2_q(N, sg_num, dim, Δs, lattice)
        @test plan isa M2QPlan

        ops = get_ops(sg_num, dim, N)
        _, shifted_ops = find_optimal_shift(ops, N)
        f_sym = make_symmetric(randn(N), shifted_ops, N)

        M = Tuple(plan.M)
        f0 = zeros(Float64, M)
        fullgrid_to_subgrid!(f0, f_sym, plan)

        f_ref = reference_diffusion_step(f_sym, Δs, lattice, N)
        f0_ref = zeros(Float64, M)
        fullgrid_to_subgrid!(f0_ref, f_ref, plan)

        f0_q = copy(f0)
        execute_m2_q!(plan, f0_q)

        err = maximum(abs.(f0_q .- f0_ref))
        @test err < 1e-10
    end

    @testset "Fm-3m (SG 225) N=$N" for N in [(16,16,16),]
        sg_num = 225
        dim = 3
        lattice = Matrix{Float64}(I, 3, 3)

        plan = plan_m2_q(N, sg_num, dim, Δs, lattice)
        @test plan isa M2QPlan

        ops = get_ops(sg_num, dim, N)
        _, shifted_ops = find_optimal_shift(ops, N)
        f_sym = make_symmetric(randn(N), shifted_ops, N)

        M = Tuple(plan.M)
        f0 = zeros(Float64, M)
        fullgrid_to_subgrid!(f0, f_sym, plan)

        f_ref = reference_diffusion_step(f_sym, Δs, lattice, N)
        f0_ref = zeros(Float64, M)
        fullgrid_to_subgrid!(f0_ref, f_ref, plan)

        f0_q = copy(f0)
        execute_m2_q!(plan, f0_q)

        err = maximum(abs.(f0_q .- f0_ref))
        @test err < 1e-10
    end

    @testset "Grid conversion" begin
        N = (16, 16, 16)
        sg_num = 47
        dim = 3
        lattice = Matrix{Float64}(I, 3, 3)
        plan = plan_m2_q(N, sg_num, dim, Δs, lattice)

        ops = get_ops(sg_num, dim, N)
        _, shifted_ops = find_optimal_shift(ops, N)
        f_sym = make_symmetric(randn(N), shifted_ops, N)

        M = Tuple(plan.M)

        # fullgrid → subgrid → fullgrid should reconstruct the symmetric field
        f0 = zeros(Float64, M)
        fullgrid_to_subgrid!(f0, f_sym, plan)

        f_expanded = zeros(Float64, N)
        subgrid_to_fullgrid!(f_expanded, f0, plan)

        # For a symmetric field, the expanded version should match the original
        err = maximum(abs.(f_expanded .- f_sym))
        @test err < 1e-14
    end
end
