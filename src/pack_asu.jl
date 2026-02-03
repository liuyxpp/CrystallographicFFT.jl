
"""
    pack_asu(points::Vector{ASUPoint}, N::Tuple, T=Float64, ArrayType=Array)

Pack a list of unique ASU points into a CrystallographicASU structure containing dense blocks.
"""
function pack_asu(points::Vector{ASUPoint}, N::Tuple, T::Type=Float64, ArrayType=Array)
    D = length(N)
    
    # 1. Group by Depth Profile
    # depth_map: depth_vector -> Vector{ASUPoint}
    depth_map = Dict{Vector{Int}, Vector{ASUPoint}}()
    for p in points
        d = p.depth
        if !haskey(depth_map, d)
            depth_map[d] = []
        end
        push!(depth_map[d], p)
    end
    
    blocks = Vector{ASUBlock}()
    
    # 2. Process each depth group
    for (depth, grouped_points) in depth_map
        # Sort points lexicographically by global index
        sort!(grouped_points, by = p->p.idx)
        
        # Determine strictness of this group
        # Identify "Active" dimensions (where depth is defined/limited or whatever)
        # But we just want to find contiguous blocks.
        
        # We greedily extract blocks
        remaining_indices = Set(p.idx for p in grouped_points)
        
        # Determine strides
        # If depth is k, stride is usually 2^k? 
        # But for GP (k=0), stride is 1? No!
        # If we split into Odd (1) and Even (0), Odd is 2k+1. Stride is 2.
        # If we split Even->Odd (2), that is 4k+2. Stride is 4.
        # Generally, stride = 2^(depth + (is_gp ? 0 : 0)).
        # Wait, depth definition:
        # 0 = Odd/GP. (Stride 2? No, could be 1 if not split).
        # Ah, depth in calc_asu increments only on Even split.
        # If we take Odd branch, depth stays same? No.
        # Logic:
        # Odd: next_N = N/2. (Stride doubles).
        # Even: next_N = N/2. (Stride doubles). next_depth += 1.
        
        # Actually, let's infer stride from data.
        # Find minimum non-zero difference in each dimension.
        
        strides = ones(Int, D)
        
        if length(grouped_points) > 1
             # Simple heuristic: Look at the first few points
             coords = hcat([p.idx for p in grouped_points]...) # D x M
             for d in 1:D
                 vals = unique(sort(coords[d, :]))
                 if length(vals) > 1
                     # Gcd of diffs
                     diffs = diff(vals)
                     s = gcd(diffs)
                     strides[d] = s
                 else
                     # Dimension is constant for this block set?
                     # strdie 1 (doesn't matter)
                     strides[d] = 1
                 end
             end
        else
            # Single point. Stride doesn't matter, set to 1.
        end
        
        # Re-sort remaining into a list to pick start
        # Use a while loop to extract blocks
        
        # Optimization: Assume for now that each depth group forms EXACTLY ONE valid grid
        # (or a set of disjoint grids with same stride). 
        # Crystallographic ASUs are usually clean.
        
        # Let's try to fit ONE block to the whole group (or connected component).
        # If that fails, we split.
        
        pool = copy(grouped_points)
        while !isempty(pool)
            p_start = pool[1]
            start_idx = p_start.idx
            
            # Flood fill or Grow Box
            # Let's simple Grow Box.
            # Find max extent in each dim matching strides.
            
            # Current box limits
            box_start = copy(start_idx)
            box_end = copy(start_idx)
            
            # Filter pool to valid candidates (on the grid defined by start_idx + k*stride)
            # candidates = [p for p in pool if (p.idx - start_idx) % stride == 0]
            
            # Grow dimension d
            # We want largest rectangular block.
            # This is "Largest Empty Rectangle" problem (but "Largest Full Rectangle").
            # Since N is small for crystallographic FFT (usually < 512), and ASU points are sparse-ish.
            
            # Simple approach:
            # 1. Extend Dim 1 as much as possible (consecutive stride steps).
            # 2. Extend Dim 2...
            
            # Construct a set for fast lookup
            pool_indices = Set(p.idx for p in pool)
            
            # Determine range
            ranges = Vector{StepRange{Int, Int}}(undef, D)
            current_shape = ones(Int, D)
            
            for d in 1:D
                # Try to extend current hyper-rectangle along dimension d
                # Current base shape is defined by ranges[1:d-1] (already fixed) and single points in d..D
                
                # Check neighbors in +d direction
                s = strides[d]
                
                # How many steps can we take?
                steps = 0
                while true
                    # specific check: for all points in current sub-block, is p + s*e_d present?
                    # Generating all points in current sub-block is expensive if large.
                    # But we build dimension by dimension.
                    # dim 1: line.
                    # dim 2: rectangle (stack of lines).
                    
                    # Iterator for current slab
                    # Ranges determined so far:
                    iter_ranges = []
                    for k in 1:(d-1)
                        push!(iter_ranges, ranges[k])
                    end
                    # Current dim and beyond are single points (start_idx)
                    for k in d:D
                        push!(iter_ranges, start_idx[k]:1:start_idx[k]) # dummy range
                    end
                    
                    # Test next slice at start_idx[d] + (steps+1)*s
                    next_val = start_idx[d] + (steps+1)*s
                    
                    # Verify slice existence
                    # Using Iterators.product on iter_ranges, but replacing d-th dim
                    slice_valid = true
                    
                    # Optimize: We only need to check the boundary?
                    # Yes.
                    
                    # Construct ranges for the slice check
                    check_ranges = copy(iter_ranges)
                    check_ranges[d] = next_val:1:next_val
                    
                    for pt in Iterators.product(check_ranges...)
                        if !(collect(pt) in pool_indices)
                            slice_valid = false
                            break
                        end
                    end
                    
                    if slice_valid
                        steps += 1
                    else
                        break
                    end
                end
                
                # Fix range for d
                ranges[d] = start_idx[d]:s:(start_idx[d] + steps*s)
                current_shape[d] = steps + 1
            end
            
            # Create Block
            # Remove used points from pool
            used_indices = []
            for pt in Iterators.product(ranges...)
                push!(used_indices, collect(pt))
            end
            
            # Verify we found something
            if isempty(used_indices)
                error("Packing algorithm failed to find any points.")
            end
            
            # Allocate Data
            # Note: We need to handle 0-based idx to 1-based array?
            # Or just store as is.
            # ASUBlock.range stores global indices.
            # ASUBlock.data stores... T=Float64 usually.
            # We initialize with zeros or undef.
            dims = tuple(length.(ranges)...)
            if ArrayType == Array
                data = zeros(T, dims)
            else
                # Fallback or CUDA support
                data = ArrayType(zeros(T, dims)) 
            end
            
            push!(blocks, ASUBlock(data, ranges, depth))
            
            # Cleanup pool
            setdiff!(pool_indices, used_indices)
            # Re-filter pool list (inefficient but safe)
            filter!(p -> (p.idx in pool_indices), pool)
        end
    end
    
    # 3. Organize into CrystallographicASU
    # Determine concrete array type A
    # We used ArrayType(zeros(T, dims)).
    # If ArrayType is Array, this produces Array{T, D}.
    # We need to construct the type A explicitely to satisfy the struct.
    # Note: size(zeros(T, dims)) -> D dims.
    
    # We can infer A from the first block's data type, or construct it.
    if isempty(blocks)
        # Empty case
        A_concrete = ArrayType{T, D} # Hope this constructor works as a type
    else
        A_concrete = typeof(blocks[1].data)
    end
    
    # Create the Dict with concrete value type
    dim_blocks = Dict{Int, Vector{ASUBlock{T, D, A_concrete}}}()
    
    for b in blocks
        eff_dim = count(r -> length(r) > 1, b.range)
        if !haskey(dim_blocks, eff_dim)
            dim_blocks[eff_dim] = ASUBlock{T, D, A_concrete}[]
        end
        # We need to convert b (type ASUBlock) to refined type? 
        # b is already ASUBlock{T, D, A_concrete} if data is correct.
        push!(dim_blocks[eff_dim], b)
    end
    
    return CrystallographicASU{D, T, A_concrete}(dim_blocks)
end
