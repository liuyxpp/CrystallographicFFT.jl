using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.SpectralIndexing
using CrystallographicFFT.ASU: ASUPoint
using Crystalline
using LinearAlgebra

@testset "Spectral Indexing" begin

    @testset "Dual Operations" begin
        # 1. Identity
        ops = [SymOp([1 0; 0 1], [0, 0])]
        dual = dual_ops(ops)
        @test dual[1].R == [1 0; 0 1]
        @test dual[1].t == [0, 0]

        # 2. 90 degree rotation (C4 in 2D)
        # R = [0 -1; 1 0]
        # R^-1 = [0 1; -1 0]
        # (R^-1)^T = [0 -1; 1 0] = R. For orthogonal matrices R = (R^-1)^T.
        R = [0 -1; 1 0]
        ops = [SymOp(R, [0, 0])]
        dual = dual_ops(ops)
        @test dual[1].R == R
        
        # 3. Simple Shear (if applicable) or non-orthogonal?
        # Let's test non-orthogonal logic if we had one. But SymOp R are integers.
        # Only meaningful for hexagonal etc. 
        # C6: R = [1 -1; 1 0] in hexagonal basis.
        # Inverse: [0 1; -1 1].
        # Transpose(Inverse): [0 -1; 1 1].
        R6 = [1 -1; 1 0]
        ops = [SymOp(R6, [0, 0])]
        dual = dual_ops(ops)
        R6_inv = [0 1; -1 1]
        @test dual[1].R == transpose(R6_inv)
    end

    @testset "Spectral ASU Completeness" begin
        # Test P1 (triclinic, |G|=1)
        # Should contain all points
        N = (4, 4)
        spec = calc_spectral_asu(1, 2, N)
        @test length(spec.points) == 16
        @test sum(p.multiplicity for p in spec.points) == 16

        # Test P2mm (Orthorhombic, |G|=4)
        # N=16x16
        N2 = (16, 16)
        spec = calc_spectral_asu(6, 2, N2)
        # Check sum of multiplicities
        @test sum(p.multiplicity for p in spec.points) == 16*16
        
        # Check that k-vectors are unique (reconstruction)
        # Or rather, that orbit expansion covers everything disjointly?
        # We rely on calc_asu correctness for that.
        # Just check get_k_vector correctness
        
        # Point at 0,0 (Gamma) should have multiplicity 1 (invariant under all R)
        # Wait, P2mm has m, m, 2. All fix 0,0.
        # So Gamma mult is 1.
        gamma_found = false
        for p in spec.points
            k = get_k_vector(spec, findfirst(x->x==p, spec.points))
            if k == [0, 0]
                gamma_found = true
                @test p.multiplicity == 1
            end
        end
        @test gamma_found
    end

    @testset "Frequency Aliasing" begin
        # N=4. Frequencies: 0, 1, -2, -1.
        # Indices: 0, 1, 2, 3.
        # 2 map to -2. 3 map to -1.
        N = (4,)
        # Fake a SpectralIndexing to test get_k_vector logic
        points = [ASUPoint([0], [], 1), ASUPoint([1], [], 1), ASUPoint([2], [], 1), ASUPoint([3], [], 1)]
        spec = SpectralIndexing(points, [], N)
        
        @test get_k_vector(spec, 1) == [0]
        @test get_k_vector(spec, 2) == [1]
        @test get_k_vector(spec, 3) == [-2]
        @test get_k_vector(spec, 4) == [-1]
        
        # N=5. Frequencies: 0, 1, 2, -2, -1.
        # Indices: 0, 1, 2, 3, 4.
        # 3 map to -2. 4 map to -1.
        N5 = (5,)
        points5 = [ASUPoint([i],[],1) for i in 0:4]
        spec5 = SpectralIndexing(points5, [], N5)
        @test get_k_vector(spec5, 1) == [0]   # 0
        @test get_k_vector(spec5, 2) == [1]   # 1
        @test get_k_vector(spec5, 3) == [2]   # 2
        @test get_k_vector(spec5, 4) == [-2]  # 3 -> -2
        @test get_k_vector(spec5, 5) == [-1]  # 4 -> -1
    end
end
