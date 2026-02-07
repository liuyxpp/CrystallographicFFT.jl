# using SubspaceConstruction # Not needed
using Test
using StaticArrays
using LinearAlgebra
using Crystalline

# We need to test internal logic, so we might need to include source or just test exported functions once implemented.
# For TDD, it is better to use the package environment.
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.ASU

@testset "Interleaved Orbit Analysis" begin
    @testset "Case P1 (No Symmetry)" begin
        N = (16, 16, 16)
        ops_p1 = get_ops(1, 3, N) # P1
        
        # L=(2,2,2) -> 8 subgrids. P1 has only identity.
        # Should result in 8 distinct orbits, each size 1.
        orbits = analyze_interleaved_orbits(N, ops_p1; L=(2,2,2))
        
        @test length(orbits) == 8
        @test all(o -> o.multiplicity == 1, orbits)
        
        # Verify coverage: Sum of multiplicity must be 8
        @test sum(o -> o.multiplicity, orbits) == 8
    end

    @testset "Case P4 (Fold-4 Rotation)" begin
        N = (16, 16, 16)
        ops_p4 = get_ops(75, 3, N) # P4
        # Rotations map (1,0,0) -> (0,1,0) -> (-1,0,0) -> (0,-1,0).
        # Modulo 2: (1,0,0) -> (0,1,0) -> (1,0,0) -> (0,1,0).
        # So (1,0,0) and (0,1,0) form an orbit of size 2.
        
        # Points in 2x2x2:
        # (0,0,0) -> (0,0,0). Size 1.
        # (0,0,1) -> (0,0,1). Size 1.
        # (1,0,0) -> (0,1,0). Size 2.
        # (1,1,0) -> (1,1,0). Size 1. (1,1 -> -1,1 -> 1,1).
        # (1,0,1) -> (0,1,1). Size 2.
        # (1,1,1) -> (1,1,1). Size 1.
        # Total orbits: 6.
        # Total points covered: 1+1+2+1+2+1 = 8.
        
        orbits = analyze_interleaved_orbits(N, ops_p4; L=(2,2,2))
        
        @test length(orbits) == 6
        @test sum(o -> o.multiplicity, orbits) == 8
        
        # Find the size-2 orbit
        size2_orbits = filter(o -> o.multiplicity == 2, orbits)
        @test length(size2_orbits) == 2
    end
    
    @testset "Case Pmmm (Mirrors)" begin
        N = (16, 16, 16)
        ops_pmmm = get_ops(47, 3, N) # Pmmm. Orthorhombic.
        # Mirrors x->-x, y->-y, z->-z.
        # Modulo 2: -1 = 1.
        # So all points map to themselves.
        # Expect 8 orbits of size 1.
        
        orbits = analyze_interleaved_orbits(N, ops_pmmm; L=(2,2,2))
        
        @test length(orbits) == 8
        @test all(o -> o.multiplicity == 1, orbits)
    end
    
    @testset "Case P4mm (Rotation + Mirror)" begin
        # Combine Rotation and Mirror.
        # P4mm (SG 99).
        # Rot(4) + Mirror(x).
        # Orbits from P4 should merge further?
        # (1,0,0) <-> (0,1,0) (Rotation)
        # (1,0,0) -> (-1,0,0) = (1,0,0) (Mirror x) -> Same.
        # So Mirror doesn't add reduction for (1,0,0) at L=2?
        # Maybe (1,1,0)?
        # (1,1,0) -> (-1,1,0) = (1,1,0). Same.
        # So P4mm behaves like P4 at L=2?
        
        N = (16, 16, 16)
        ops_p4mm = get_ops(99, 3, N)
        orbits = analyze_interleaved_orbits(N, ops_p4mm; L=(2,2,2))
        
        # Check if reduction is better than P4
        @test length(orbits) <= 6
    end
end
