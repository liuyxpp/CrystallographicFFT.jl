using Test
using CrystallographicFFT
using CrystallographicFFT.DiffusionSolver
using LinearAlgebra

@testset "Diffusion Solver" begin
    # 1. Setup P2mm 2D case
    N = (8, 8)
    L = 1.0
    lattice = [L 0; 0 L]
    sg_num = 6 # P2mm
    dim = 2
    Δs = 0.1
    
    solver = plan_diffusion(N, lattice, sg_num, dim, Δs)
    
    @test solver isa MatrixDiffusionSolver
    n_real = length(solver.real_asu)
    n_spec = length(solver.spec_asu.points)
    
    # 2. Check Matrix Invertibility
    # M_inv * M should be Identity (if n_real == n_spec and basis is good)
    # For P2mm, real ASU and spectral ASU size should match exactly?
    # 8x8 = 64. P2mm |G|=4. ASU ~ 16.
    # Check dimensions
    println("Size Real: $n_real, Size Spec: $n_spec")
    
    I_check = solver.M_inv * solver.M
    err = norm(I_check - I(n_spec))
    println("Identity Check Error: $err")
    @test isapprox(I_check, I(n_spec), atol=1e-10)
    
    # 3. Check Real Value Preservation
    # If u is real, M * M_inv * u should be real
    u_real_in = rand(n_real)
    u_spec = solver.M_inv * u_real_in
    u_real_out = solver.M * u_spec
    
    @test maximum(abs.(imag.(u_real_out))) < 1e-10
    @test isapprox(real.(u_real_out), u_real_in, atol=1e-10)
    
    # 4. Check Step execution
    w_real = rand(n_real)
    u_copy = copy(u_real_in)
    step_diffusion!(solver, u_copy, w_real, 1.0)
    
    # Check something changed
    @test u_copy != u_real_in
    @test length(u_copy) == n_real
end

@testset "Q-Fused DiffusionSolver" begin
    using FFTW
    using CrystallographicFFT.SymmetryOps: get_ops
    using CrystallographicFFT.ASU: find_optimal_shift
    using CrystallographicFFT.QFusedKRFFT: fullgrid_to_subgrid!

    """Full-grid FFT → K → IFFT reference"""
    function ref_diffusion(f_sym, Δs, lattice, N)
        D = length(N)
        recip_B = 2π * inv(lattice)'
        F = fft(complex(f_sym))
        for ci in CartesianIndices(F)
            h = [ci[d]-1 for d in 1:D]
            for d in 1:D; h[d] >= N[d]÷2 && (h[d] -= N[d]); end
            Kvec = recip_B * h
            F[ci] *= exp(-dot(Kvec, Kvec) * Δs)
        end
        return real.(ifft(F))
    end

    """Build symmetrized field"""
    function make_sym(ops, N)
        f = randn(N)
        f_sym = zeros(N)
        for ci in CartesianIndices(f_sym)
            x = [ci[d]-1 for d in 1:length(N)]
            for op in ops
                xr = mod.(Int.(op.R) * x .+ round.(Int, op.t), collect(N))
                f_sym[ci] += f[xr[1]+1, xr[2]+1, xr[3]+1]
            end
        end
        f_sym ./= length(ops)
    end

    @testset "Pmmm (SG 47) N=$N" for N in [(16,16,16), (32,32,32)]
        Δs = 0.1
        lattice = Matrix{Float64}(I, 3, 3)
        sg_num = 47

        qsolver = plan_diffusion(N, lattice, sg_num, 3, Δs; method=:qfused)
        @test qsolver isa QFusedDiffusionSolver

        _, shifted_ops = find_optimal_shift(get_ops(sg_num, 3, N), N)
        f_sym = make_sym(shifted_ops, N)

        M = Tuple(qsolver.plan.M)
        f0 = zeros(Float64, M)
        fullgrid_to_subgrid!(f0, f_sym, qsolver.plan)

        f_ref = ref_diffusion(f_sym, Δs, lattice, N)
        f0_ref = zeros(Float64, M)
        fullgrid_to_subgrid!(f0_ref, f_ref, qsolver.plan)

        f0_test = copy(f0)
        apply_diffusion_operator!(qsolver, f0_test)

        @test maximum(abs.(f0_test .- f0_ref)) < 1e-10
    end

    @testset "P4/mmm (SG 123) N=(16,16,16)" begin
        N = (16, 16, 16)
        Δs = 0.05
        lattice = [1.0 0 0; 0 1.0 0; 0 0 1.5]
        sg_num = 123

        qsolver = plan_diffusion(N, lattice, sg_num, 3, Δs; method=:qfused)
        @test qsolver isa QFusedDiffusionSolver

        _, shifted_ops = find_optimal_shift(get_ops(sg_num, 3, N), N)
        f_sym = make_sym(shifted_ops, N)

        M = Tuple(qsolver.plan.M)
        f0 = zeros(Float64, M)
        fullgrid_to_subgrid!(f0, f_sym, qsolver.plan)

        f_ref = ref_diffusion(f_sym, Δs, lattice, N)
        f0_ref = zeros(Float64, M)
        fullgrid_to_subgrid!(f0_ref, f_ref, qsolver.plan)

        f0_test = copy(f0)
        apply_diffusion_operator!(qsolver, f0_test)

        @test maximum(abs.(f0_test .- f0_ref)) < 1e-10
    end
end

