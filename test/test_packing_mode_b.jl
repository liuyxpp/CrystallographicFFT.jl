
using Test
using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using StaticArrays

@testset "Interleaved Packing (Mode B)" begin
    N = (16, 16, 16)
    u = rand(ComplexF64, N)
    
    @testset "P4 case" begin
        ops = get_ops(75, 3, N) # P4
        asu = pack_asu_interleaved(u, N, ops; L=(2,2,2))
        
        # Check type
        @test asu isa CrystallographicASU
        
        # Check blocks
        # Should be 3D blocks (key = 3)
        @test haskey(asu.dim_blocks, 3)
        blocks = asu.dim_blocks[3]
        
        # P4 should have 6 independent subgrids
        @test length(blocks) == 6
        
        # Check first block
        b1 = blocks[1]
        @test b1.orbit !== nothing
        
        # Verify data matches range
        # Ranges are StepRange
        # u[b1.range...] performs strided copy
        extracted = u[b1.range...]
        @test b1.data == extracted
        
        # Verify orbit mapping
        # b1.orbit.representative is 0-based shift
        # b1.range starts at rep+1
        rep = b1.orbit.representative
        range_start = [first(r) for r in b1.range]
        @test range_start == rep .+ 1
        
        # Verify steps
        range_step = [step(r) for r in b1.range]
        @test all(s -> s == 2, range_step)
    end
    
    @testset "Pmmm case" begin
        ops = get_ops(47, 3, N) # Pmmm. Orthorhombic.
        asu = pack_asu_interleaved(u, N, ops; L=(2,2,2))
        
        blocks = asu.dim_blocks[3]
        # Pmmm with L=2 should produce 8 blocks (no reduction)
        @test length(blocks) == 8
        
        # All blocks should be independent
        @test all(b -> b.orbit.multiplicity == 1, blocks)
        
        # Total data volume check
        # Each block is N/2 size -> N^3 / 8 points
        # 8 blocks * (N^3/8) = N^3 points = full grid
        total_points = sum(length(b.data) for b in blocks)
        @test total_points == prod(N)
    end
end
