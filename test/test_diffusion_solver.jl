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
    
    @test solver isa DiffusionSolver
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
