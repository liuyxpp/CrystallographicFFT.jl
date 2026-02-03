using CrystallographicFFT.ASU

function verify_pack(sg_num, dim, N, name)
    println("-"^40)
    println("Packing Test: $name (SG $sg_num, $dim-D, $N)")
    
    # Calculate points
    ops = get_ops(sg_num, dim, N)
    points = calc_asu(N, ops)
    println("Total Points: $(length(points))")
    
    # Pack
    c_asu = pack_asu(points, N)
    
    total_elements = 0
    
    for (eff_dim, blocks) in c_asu.dim_blocks
        println("Dimension $eff_dim: $(length(blocks)) blocks")
        for (i, b) in enumerate(blocks)
            sz = size(b.data)
            range_str = join(b.range, " × ")
            # println("  Block $i: Size $sz, Range $range_str, Depth $(b.depth)")
            println("  Block $i: Size $sz")
            total_elements += length(b.data)
            
            # Verify data type
            @assert eltype(b.data) == Float64
            @assert b.data isa Array
        end
    end
    
    println("Total Elements in Blocks: $total_elements")
    
    if total_elements == length(points)
        println("✅ Count Match!")
    else
        println("❌ Count Mismatch! Expected $(length(points)), Got $total_elements")
    end
end

# Test p2mm (should have nice blocks)
verify_pack(6, 2, (8, 8), "p2mm")

# Test p2mg (should handle coupled column x=0 correctly)
# x=0 col is size 5. It is 0:1:0 x 0:1:4. 
# It should be packed as one 1D block.
verify_pack(7, 2, (8, 8), "p2mg")
