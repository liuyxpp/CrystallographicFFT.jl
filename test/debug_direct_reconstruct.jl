using CrystallographicFFT
using CrystallographicFFT.ASU
using FFTW
using LinearAlgebra
using Statistics

function expand_to_full_grid(asu::CrystallographicASU{D, T, A}, N::Tuple, ops::Vector{SymOp}) where {D, T, A}
    full_grid = zeros(ComplexF64, N)
    
    for (d, blocks) in asu.dim_blocks
        for block in blocks
            cis = CartesianIndices(size(block.data))
            for index in cis
                val = block.data[index]
                g_idx_0based = [block.range[k][index[k]] for k in 1:D]
                orbit = Set{Vector{Int}}()
                stack = [g_idx_0based]
                push!(orbit, g_idx_0based)
                while !isempty(stack)
                    curr = pop!(stack)
                    for op in ops
                        next_p = apply_op(op, curr, N)
                        if !(next_p in orbit)
                            push!(orbit, next_p)
                            push!(stack, next_p)
                        end
                    end
                end
                for p in orbit
                    full_grid[(p .+ 1)...] = val
                end
            end
        end
    end
    return full_grid
end

function expand_to_full_grid_summed(asu::CrystallographicASU{D, T, A}, N::Tuple, ops::Vector{SymOp}) where {D, T, A}
    full_grid = zeros(ComplexF64, N)
    for (d, blocks) in asu.dim_blocks
        for block in blocks
            cis = CartesianIndices(size(block.data))
            for index in cis
                val = block.data[index]
                g_idx_0based = [block.range[k][index[k]] for k in 1:D]
                for op in ops
                    p = apply_op(op, g_idx_0based, N)
                    full_grid[(p .+ 1)...] += val
                end
            end
        end
    end
    return full_grid
end


# Copy of the function to debug
function direct_reconstruct_spectrum(spectral_asu, N, ops)
    D = length(N)
    F_recon = zeros(ComplexF64, N)
    
    # Pre-compute block metadata
    block_meta = []
    for (d, blocks) in spectral_asu.dim_blocks
        for block in blocks
            sizes = size(block.data)
            starts = [first(r) for r in block.range] # 0-based start
            steps = [step(r) for r in block.range]
            push!(block_meta, (data=block.data, sizes=sizes, starts=starts, steps=steps))
        end
    end

    cis = CartesianIndices(N)
    for idx in cis
        h = collect(Tuple(idx)) .- 1 
        
        val_h = 0.0 + 0.0im
        
        for op in ops
            # 1. Phase Shift (Symmetry)
            phase_arg_g = sum(h[i] * op.t[i] / N[i] for i in 1:D)
            phase_g = exp(-2π * im * phase_arg_g)
            
            # 2. Rotated k
            k_rot = transpose(op.R) * h
            
            # 3. Fetch Y
            Y_val = 0.0 + 0.0im
            for b in block_meta
                # A. Block Phase Shift
                phase_arg_b = sum(k_rot[i] * b.starts[i] / N[i] for i in 1:D)
                phase_b = exp(-2π * im * phase_arg_b)
                
                # B. Lookup
                idx_local = zeros(Int, D)
                for i in 1:D
                    sz = b.sizes[i]
                    idx_local[i] = mod(k_rot[i], sz) + 1
                end
                
                block_val = b.data[idx_local...]
                Y_val += phase_b * block_val
            end
            
            val_h += phase_g * Y_val
        end
        F_recon[idx] = val_h
    end
    return F_recon
end

function debug_delta()
    sg_num = 6 # p2mm
    N = (16, 16) # Small grid for debug
    dim = 2
    D = length(N)
    
    plan = plan_cfft(N, sg_num, ComplexF64, Array)
    input_asu = deepcopy(plan.asu)
    
    # Set Delta function at (0,0)
    # The point (0,0) is in the first block (starts=[0,0])
    # Let's locate it.
    found = false
    for (d, blocks) in input_asu.dim_blocks
        for b in blocks
            cis = CartesianIndices(size(b.data))
            for idx in cis
                g_idx = [b.range[k][idx[k]] for k in 1:D]
                if all(g_idx .== 0)
                    b.data[idx] = 1.0
                    found = true
                    println("Placed delta at global $(g_idx) in block with start $([first(r) for r in b.range])")
                else
                    b.data[idx] = 0.0
                end
            end
        end
    end
    
    spectral_asu = deepcopy(plan.asu)
    mul!(spectral_asu, plan, input_asu)
    
    ops = get_ops(sg_num, dim, N)
    println("Ops: ", ops)

"""
    direct_reconstruct_spectrum_weighted(spectral_asu::CrystallographicASU{D, T, A}, N::Tuple, ops::Vector{SymOp})

Directly compute the Full Grid Spectrum F(h) with Multiplicity Correction.
Weights each block's contribution by 1/|Stabilizer| to correct for overcounting of special positions
inherent in the Sum-Over-Group formula.
"""
function direct_reconstruct_spectrum_weighted(spectral_asu::CrystallographicASU{D, T, A}, N::Tuple, ops::Vector{SymOp}) where {D, T, A}
    F_recon = zeros(ComplexF64, N)
    
    # Pre-compute block metadata and weights
    block_meta = []
    for (d, blocks) in spectral_asu.dim_blocks
        for block in blocks
            sizes = size(block.data)
            starts = [first(r) for r in block.range] 
            steps = [step(r) for r in block.range]
            
            # Compute Stabilizer Order (using start point)
            stab_count = 0
            for op in ops
                p = apply_op(op, starts, N)
                if p == starts
                    stab_count += 1
                end
            end
            weight = 1.0 / stab_count
            
            push!(block_meta, (data=block.data, sizes=sizes, starts=starts, steps=steps, weight=weight))
        end
    end

    cis = CartesianIndices(N)
    for idx in cis
        h = collect(Tuple(idx)) .- 1 
        
        val_h = 0.0 + 0.0im
        
        for op in ops
            phase_arg_g = sum(h[i] * op.t[i] / N[i] for i in 1:D)
            phase_g = exp(-2π * im * phase_arg_g)
            
            k_rot = transpose(op.R) * h
            
            Y_val = 0.0 + 0.0im
            for b in block_meta
                # Same Y_block logic
                phase_arg_b = sum(k_rot[i] * b.starts[i] / N[i] for i in 1:D)
                phase_b = exp(-2π * im * phase_arg_b)
                
                idx_local = zeros(Int, D)
                for i in 1:D
                    sz = b.sizes[i]
                    idx_local[i] = mod(k_rot[i], sz) + 1
                end
                
                block_val = b.data[idx_local...]
                
                # Apply Weight here
                Y_val += phase_b * block_val * b.weight
            end
            
            val_h += phase_g * Y_val
        end
        
        F_recon[idx] = val_h
    end
    
    return F_recon
end

    println("Analyzing Block Stabilizers...")
    for (d, blocks) in input_asu.dim_blocks
        for (i, block) in enumerate(blocks)
            start_idx = [first(r) for r in block.range]
            # Check stabilizer of start index
            stab_count = 0
            for op in ops
                p = apply_op(op, start_idx, N)
                if p == start_idx
                    stab_count += 1
                end
            end
            println("  Block $d-$i Range: $(block.range) | Stab Order: $stab_count")
            
            # Check if all points in block have same stabilizer?
            # Probe end point
            end_idx = [last(r) for r in block.range]
            stab_count_end = 0
            for op in ops
                p = apply_op(op, end_idx, N)
                if p == end_idx
                    stab_count_end += 1
                end
            end
            if stab_count != stab_count_end
                println("    WARNING: Block has mixed stabilizers! Start: $stab_count, End: $stab_count_end")
            end
            
            # Additional probe for mid point if size > 1 in both dims
             if all(size(block.data) .> 1)
                mid_idx = [block.range[k][2] for k in 1:D]
                stab_count_mid = 0
                for op in ops
                    p = apply_op(op, mid_idx, N)
                    if p == mid_idx
                        stab_count_mid += 1
                    end
                end
                if stab_count != stab_count_mid
                    println("    WARNING: Block has mixed stabilizers (Mid)! Mid: $stab_count_mid")
                end
             end
        end
    end
    
    # FULL GRID REFERENCE
    # Delta at (0,0) -> F(h) = 1 everywhere.
    
    recon = direct_reconstruct_spectrum(spectral_asu, N, ops)
    
    println("Recon at (0,0): ", recon[1,1])
    println("Recon at (1,0): ", recon[2,1])
    println("Recon at (0,1): ", recon[1,2])
    
    # Check if constant
    is_const = all(y -> isapprox(y, recon[1,1], atol=1e-10), recon)
    println("Is constant? $is_const")
    
    if !is_const
        println("Recon Matrix:")
        display(recon)
    end
    
    scale = abs(recon[1,1])
    println("Scale factor (expected ~1.0?): $scale")
    
    # Random Test
    println("\nRandom Data Test:")
    for (d, blocks) in input_asu.dim_blocks
        for b in blocks
            b.data .= rand(ComplexF64, size(b.data))
        end 
    end
    
    # Random Test with SUMMED
    println("\nRandom Data Test (SUMMED):")
    # Regenerate random
    for (d, blocks) in input_asu.dim_blocks
        for b in blocks
            b.data .= rand(ComplexF64, size(b.data))
        end 
    end
    

    
    # Recalculate Ref for Split Test
    println("Splitting ASU into Homogeneous Blocks...")
    split_asu = split_homogeneous_blocks(input_asu, N, ops)
    full_data_unique = expand_to_full_grid(input_asu, N, ops)
    fft_ref_unique = fft(full_data_unique)
    
    # Pseudo-Spectral ASU
    spectral_blocks_split = []
    
    # Compute Spectral ASU for split blocks Manually (emulate CFFT)
    # We create a pseudo-ASU structure
    dim_blocks_split = Dict{Int, Vector{Any}}() # Use Any to hold named tuples or structs
    # But direct_reconstruct checks .dim_blocks
    # So we need to put them back into a structure compatible with the function.
    # We can perform the FFT here and pass "spectral blocks".
    
    # Actually, let's create a Helper Structure or reuse CrystalASU
    # reusing CrystalASU is hard because types.
    # Let's just adjust `direct_reconstruct_spectrum_weighted` to accept a list of blocks.
    
    spectral_blocks_split = []
    
    for (d, blocks) in split_asu.dim_blocks
        for b in blocks
            # FFT of block
            b_spec_data = fft(b.data)
            push!(spectral_blocks_split, (data=b_spec_data, range=b.range))
        end
    end
    
    # We need to adapt direct_reconstruct to use this list
    # I will modify the function to accept `blocks_list` instead of `asu`.
    
    recon_split = direct_reconstruct_from_list(spectral_blocks_split, N, ops)
    
    scale_w = mean(abs.(recon_split)) / mean(abs.(fft_ref_unique))
    println("Split-Weighted Scale Factor: $scale_w")
    
    diff_w = norm(recon_split - fft_ref_unique) / norm(fft_ref_unique)
    println("Split-Weighted Diff: $diff_w")
    
    if diff_w > 1e-10
        println("Split Recon Failure")
        display(recon_split[1:2, 1:2])
        display(fft_ref_unique[1:2, 1:2])
    end
end

function split_homogeneous_blocks(asu, N, ops)
    # Returns new ASU structure (or dict) with split blocks
    # For debug, we return a mock object with .dim_blocks
    new_blocks = Dict{Int, Vector{Any}}()
    
    for (d, blocks) in asu.dim_blocks
        for block in blocks
            # Scan all points, group by stabilizer order
            # Since blocks are multidimensional, we iterate CartesianIndices
            # If we find mixed stabilizers, we try to carve out sub-blocks.
            # Simple heuristic: Split 1D ranges if possible.
            # But general n-D splitting is hard.
            # For p2mm, we expect 1D or 0D blocks.
            # So we just iterate points and form 1x1 blocks for everything? (Inefficient but robust for verification).
            # Yes! For Verification, efficiency doesn't matter.
            # Split into 1x1 blocks (Points).
            
            cis = CartesianIndices(size(block.data))
            for index in cis
                val = block.data[index]
                g_idx = [block.range[k][index[k]] for k in 1:length(N)]
                
                # Create 1x1 block
                new_start = g_idx
                # create range: start:1:start
                new_range = Tuple(s:1:s for s in new_start)
                new_data = reshape([val], (1,1)[1:length(N)]) # 2D -> 1x1
                new_data = fill(ComplexF64(val), Tuple(ones(Int, length(N))))
                
                # Approximate D key?
                # D=0 (point).
                if !haskey(new_blocks, 0)
                    new_blocks[0] = []
                end
                push!(new_blocks[0], (data=new_data, range=new_range))
            end
        end
    end
    return (dim_blocks=new_blocks,)
end

function direct_reconstruct_from_list(blocks_list, N, ops)
    D = length(N)
    F_recon = zeros(ComplexF64, N)
    
    # Pre-compute meta
    block_meta = []
    for b in blocks_list
        sizes = size(b.data)
        starts = [first(r) for r in b.range]
        # steps = [step(r) for r in b.range] # All 1
        
        # Calc Weight
        stab_count = 0
        for op in ops
            p = apply_op(op, starts, N)
            if p == starts
                stab_count += 1
            end
        end
        weight = 1.0 / stab_count
        
        push!(block_meta, (data=b.data, sizes=sizes, starts=starts, weight=weight))
    end
    
    cis = CartesianIndices(N)
    for idx in cis
        h = collect(Tuple(idx)) .- 1 
        val_h = 0.0 + 0.0im
        for op in ops
            phase_arg_g = sum(h[i] * op.t[i] / N[i] for i in 1:D)
            phase_g = exp(-2π * im * phase_arg_g)
            k_rot = transpose(op.R) * h
            
            Y_val = 0.0 + 0.0im
            for b in block_meta
                 phase_arg_b = sum(k_rot[i] * b.starts[i] / N[i] for i in 1:D)
                 phase_b = exp(-2π * im * phase_arg_b)
                 # idx_local always 1 for 1x1
                 block_val = b.data[1] 
                 Y_val += phase_b * block_val * b.weight
            end
            val_h += phase_g * Y_val
        end
        F_recon[idx] = val_h
    end
    return F_recon
end

debug_delta()
