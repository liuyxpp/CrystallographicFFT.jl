using CrystallographicFFT
using CrystallographicFFT.ASU
using FFTW
using LinearAlgebra
using Random
using Test
using Statistics

"""
    expand_to_full_grid(asu::CrystallographicASU, N::Tuple, ops::Vector{SymOp})

Expand the Real Space ASU data to the full real space grid using symmetry operations.
This uses the property f(Rx+t) = f(x).
"""
function expand_to_full_grid(asu::CrystallographicASU{D, T, A}, N::Tuple, ops::Vector{SymOp}) where {D, T, A}
    full_grid = zeros(ComplexF64, N)
    
    for (d, blocks) in asu.dim_blocks
        for block in blocks
            cis = CartesianIndices(size(block.data))
            for index in cis
                val = block.data[index]
                
                # Convert to global coords (0-based)
                g_idx_0based = [block.range[k][index[k]] for k in 1:D]
                
                # Generate Orbit
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
                
                # Fill full grid
                for p in orbit
                    full_grid[(p .+ 1)...] = val
                end
            end
        end
    end
    
    return full_grid
end

"""
    direct_reconstruct_spectrum_weighted(spectral_asu, N::Tuple, ops::Vector{SymOp})

Directly compute the Full Grid Spectrum F(h) with Multiplicity Correction.
Weights each block's contribution by 1/|Stabilizer| to correct for overcounting of special positions.
This requires that each block in `spectral_asu` is homogeneous (all points have same stabilizer order).
"""
function direct_reconstruct_spectrum_weighted(spectral_asu, N::Tuple, ops::Vector{SymOp})
    D = length(N)
    F_recon = zeros(ComplexF64, N)
    
    # Pre-compute block metadata and weights
    block_meta = []
    # Use generic iteration if spectral_asu is a Dict or struct
    # We assume spectral_asu has .dim_blocks or is a Dict (if passed directly)
    iter_source = hasproperty(spectral_asu, :dim_blocks) ? spectral_asu.dim_blocks : spectral_asu

    for (d, blocks) in iter_source
        for block in blocks
            sizes = size(block.data)
            starts = [first(r) for r in block.range] 
            steps = [step(r) for r in block.range]
            
            # Compute Stabilizer Order of the block start point
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

    # Iterate over full reciprocal grid h
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
                
                idx_local = zeros(Int, D)
                for i in 1:D
                    sz = b.sizes[i]
                    idx_local[i] = mod(k_rot[i], sz) + 1
                end
                
                # For 1x1 blocks (from split), idx_local is always 1.
                # For larger blocks, use calculated index.
                block_val = b.data[idx_local...]
                
                Y_val += phase_b * block_val * b.weight
            end
            val_h += phase_g * Y_val
        end
        F_recon[idx] = val_h
    end
    
    return F_recon
end

"""
    split_homogeneous_blocks(asu, N)

Split the ASU blocks into minimal homogeneous sub-blocks (points) ensuring that
each block contains points with uniform stabilizer order.
Required for Direct Verification of grids with mixed-stabilizer blocks (e.g. p2mm N=16).
"""
function split_homogeneous_blocks(asu::CrystallographicASU{D, T, A}, N::Tuple) where {D, T, A}
    # Returns a structure compatible with direct_reconstruct_spectrum_weighted
    # Dict{Int, Vector{Any}}
    new_blocks = Dict{Int, Vector{Any}}()
    
    for (d, blocks) in asu.dim_blocks
        for block in blocks
            # Decompose into 1x1 blocks for absolute safety
            cis = CartesianIndices(size(block.data))
            for index in cis
                val = block.data[index]
                g_idx = [block.range[k][index[k]] for k in 1:D]
                
                new_start = g_idx
                # create range: start:1:start
                new_range = Tuple(s:1:s for s in new_start)
                new_data = fill(ComplexF64(val), Tuple(ones(Int, D)))
                
                # Key 0 for points
                if !haskey(new_blocks, 0)
                    new_blocks[0] = []
                end
                
                # Mock block object (needs .data and .range)
                push!(new_blocks[0], (data=new_data, range=new_range))
            end
        end
    end
    return (dim_blocks=new_blocks,)
end


"""
    verify_spectral_consistency(full_grid, cfft_asu, N)

Verify that the CFFT blocks correspond to the correctly aliased and phase-shifted
slices of the Full FFT (Eq 16 in Design Doc).
"""
function verify_spectral_consistency(full_grid, cfft_asu, N)
    D = length(N)
    full_spec = fft(full_grid)
    
    all_match = true
    
    for (dim_key, blocks) in cfft_asu.dim_blocks
        for (b_idx, block) in enumerate(blocks)
            sizes = size(block.data) 
            starts = [first(r) for r in block.range]
            range_steps = [step(r) for r in block.range]
            
            p_implied = range_steps
            total_span = sizes .* p_implied
            
            is_partial = any(total_span .!= N)
            
            if is_partial
                 continue
            end
            
            # Process Full Period Blocks
            steps = p_implied 
            expected = zeros(ComplexF64, sizes)
            cis = CartesianIndices(sizes)
            
            for idx in cis
                idx_tuple = Tuple(idx)
                q = idx_tuple .- 1 
                
                sum_val = 0.0 + 0.0im
                
                # Sum over aliased frequencies
                r_iterators = [0:(steps[d]-1) for d in 1:D]
                
                for r_tuple in Iterators.product(r_iterators...)
                    r = collect(r_tuple)
                    k = [q[d] + r[d] * sizes[d] for d in 1:D]
                    
                    # Compute Phase Shift
                    phase_arg = sum(k[d] * starts[d] / N[d] for d in 1:D)
                    phase = exp(2π * im * phase_arg)
                    
                    # Fetch Full FFT value
                    k_idx = [mod(k[d], N[d]) + 1 for d in 1:D] 
                    val_full = full_spec[k_idx...]
                    sum_val += val_full * phase
                end
                
                # Normalization for aliasing summation
                P = prod(steps)
                sum_val /= P
                expected[idx] = sum_val
            end
            
            diff = norm(block.data - expected) / (norm(expected) + 1e-12)
            if diff > 1e-10
                println("    Block $dim_key-$b_idx: Spectral Mismatch! RelErr = $diff")
                all_match = false
            end
        end
    end
    return all_match
end

function test_p2mm_verification()
    println("="^60)
    println("Verifying CFFT against FFTW for p2mm (2D)...")
    
    sg_num = 6 # p2mm
    N = (16, 16) 
    dim = 2
    
    # 1. Plan CFFT
    plan = plan_cfft(N, sg_num, ComplexF64, Array)
    println("  Plan created. $(length(plan.fft_plans)) unique FFT plans.")
    
    # 2. Create Random Data on ASU
    println("  Generating random ASU data...")
    input_asu = deepcopy(plan.asu)
    for (d, blocks) in input_asu.dim_blocks
        for b in blocks
            b.data .= rand(ComplexF64, size(b.data))
        end
    end
    
    # 3. Expand to Full Grid
    ops = get_ops(sg_num, dim, N)
    full_data = expand_to_full_grid(input_asu, N, ops)
    println("  Expanded to full grid (Size $(size(full_data))).")
    
    # Verify Symmetry
    is_symmetric = true
    for idx_tuple in CartesianIndices(full_data)
        idx = collect(Tuple(idx_tuple)) .- 1
        val = full_data[idx_tuple]
        for op in ops
            next_p = apply_op(op, idx, N)
            val_sym = full_data[(next_p .+ 1)...]
            if !isapprox(val, val_sym, atol=1e-10)
                global is_symmetric = false
                break
            end
        end
        if !is_symmetric break end
    end
    println("  Full grid symmetry check: $(is_symmetric ? "PASS" : "FAIL")")
    @test is_symmetric
    
    # 4. Run CFFT (Forward)
    spectral_asu = deepcopy(plan.asu) 
    mul!(spectral_asu, plan, input_asu)
    println("  Ran CFFT Forward.")
    
    # 5. Verify Spectral Consistency (Aliasing)
    println("  Verifying Spectral Consistency (Aliasing & Phase check)...")
    consistent = verify_spectral_consistency(full_data, spectral_asu, N)
    println("  Spectral Verification: $(consistent ? "PASS" : "FAIL")")
    @test consistent

    # 6. Roundtrip Verification (Inverse using ldiv!)
    recon_asu = deepcopy(plan.asu)
    ldiv!(recon_asu, plan, spectral_asu)
    
    max_diff = 0.0
    for (d, blocks) in recon_asu.dim_blocks
        for (i, b_recon) in enumerate(blocks)
            b_in = input_asu.dim_blocks[d][i]
            diff = norm(b_recon.data - b_in.data)
            max_diff = max(max_diff, diff)
        end
    end
    println("  Roundtrip Max Error: $max_diff")
    @test max_diff < 1e-10
    
    println("  Verifying Full Spectral Reconstruction (Direct Weighted Method)...")
    
    # A. Reference FFT from Full Data (Unique)
    # This is the physical ground truth
    fft_ref = fft(full_data)
    
    # B. Helper: Split Real ASU into homogeneous blocks for verification
    # Note: We must emulate the spectral blocks that WOULD result from homogeneous separation.
    split_asu = split_homogeneous_blocks(input_asu, N)
    
    # C. Compute Spectral ASU for split blocks Manually
    split_spectral_blocks = Dict{Int, Vector{Any}}()
    for (d, blocks) in split_asu.dim_blocks
        split_spectral_blocks[d] = []
        for b in blocks
            # FFT of block data. For 1x1 block, it's just the value.
            b_spec_data = fft(b.data)
            push!(split_spectral_blocks[d], (data=b_spec_data, range=b.range))
        end
    end
    split_spectral_asu = (dim_blocks=split_spectral_blocks,)
    
    # D. Reconstruct from Split Spectral ASU using Direct Weighted Method
    recon_spec = direct_reconstruct_spectrum_weighted(split_spectral_asu, N, ops)
    
    # E. Compare
    spec_diff = norm(recon_spec - fft_ref) / norm(fft_ref)
    println("  Spectral Reconstruction Error: $spec_diff")
    @test spec_diff < 1e-10
    
    println("="^60)
end

test_p2mm_verification()
