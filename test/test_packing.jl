using Test
using CrystallographicFFT.ASU
using Crystalline

@testset "ASU Packing" begin
    
    function verify_pack(sg_num, dim, N)
        points, shift = calc_asu(sg_num, dim, N)
        total_points = length(points)
        
        c_asu = pack_asu(points, N; shift=shift)
        
        total_packed = 0
        block_count = 0
        
        for (d, blocks) in c_asu.dim_blocks
            for b in blocks
                total_packed += length(b.data)
                block_count += 1
            end
        end
        
        @test total_packed == total_points
        return block_count
    end

    @testset "p2mm (2D)" begin
        # 8x8. Expect 16 points. Ideally 1 block (4x4).
        count = verify_pack(6, 2, (8, 8))
    end

    @testset "p2mg (2D)" begin
        verify_pack(7, 2, (8, 8))
    end
    
    @testset "Pmmm (3D)" begin
        verify_pack(47, 3, (8, 8, 8))
    end
end
