using Test
using CrystallographicFFT
using CrystallographicFFT.MatrixQ
using CrystallographicFFT.SpectralIndexing
using CrystallographicFFT.SymmetryOps
using LinearAlgebra

@testset "Matrix Q Generation" begin
    
    # 1. Setup a simple cubic lattice
    # P1 symmetry (no reduction, full grid check)
    N = (4, 4, 4)
    ops_p1 = get_ops(1, 3, N) # P1
    # Spectral ASU for P1 should be full grid (64 pts) - technically
    # But wait, calc_asu for P1 gives full grid? Yes.
    
    spec_indexing = calc_spectral_asu(1, 3, N) 
    
    L = 10.0
    lattice = [L 0 0; 0 L 0; 0 0 L] # Cubic box L=10
    Δs = 0.01
    
    # Calculate Kernel Function
    kernel_func = calc_gradient_term(N, Δs, lattice)
    
    # 2. Check Kernel Function Values manually
    # k = (1, 0, 0)
    # b1 = (2π/L, 0, 0)
    # K = (2π/L, 0, 0)
    # k^2 = (2π/L)^2
    # val = exp(-(2π/L)^2 * 0.01)
    k_vec = [1, 0, 0]
    expected_val = exp(- (2*π/L)^2 * 0.01 )
    @test isapprox(kernel_func(k_vec), expected_val)

    # 3. Generate Matrix Q
    Q = calc_matrix_q(spec_indexing, kernel_func)
    
    @test size(Q) == (64, 64)
    @test isdiag(Q)
    
    # Check a specific value in Q
    # Find index corresponding to k=[1,0,0]
    # In P1 ASU (unpacked), ordering depends on iteration.
    # Let's search.
    idx_100 = 0
    for i in 1:64
        if get_k_vector(spec_indexing, i) == [1, 0, 0]
            idx_100 = i
            break
        end
    end
    
    @test idx_100 > 0
    @test isapprox(Q[idx_100, idx_100], expected_val)
    
    # 4. Symmetry Check (P2mm)
    # N = (8, 8)
    # Lattice = Square L=1
    N2 = (8, 8)
    lattice2 = [1.0 0; 0 1.0]
    spec_p2mm = calc_spectral_asu(6, 2, N2) # P2mm
    kernel_func2 = calc_gradient_term(N2, 0.1, lattice2)
    Q_p2mm = calc_matrix_q(spec_p2mm, kernel_func2)
    
    # Check dimension: P2mm reduces N^2 approx by 4. 64/4 = 16.
    # Exact count for 8x8 should be checked.
    # 0,0 (1)
    # 4,4 (1)
    # others...
    n_asu = length(spec_p2mm.points)
    @test size(Q_p2mm) == (n_asu, n_asu)
    
    # Verify values are appropriate (<= 1.0 for diffusion)
    @test all(Q_p2mm[i,i] <= 1.0 + 1e-10 for i in 1:n_asu)
    @test all(Q_p2mm[i,i] >= 0.0 for i in 1:n_asu)
end
