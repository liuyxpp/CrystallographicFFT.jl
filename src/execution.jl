
import LinearAlgebra: ldiv!, mul!

"""
    mul!(out::CrystallographicASU, plan::CFFTPlan, in::CrystallographicASU)

Perform the forward Crystallographic FFT (Spatial -> Spectral).
Applies the sub-plans to each block.
"""
function mul!(out::CrystallographicASU{D, To, Ao}, plan::CFFTPlan, in::CrystallographicASU{D, Ti, Ai}) where {D, To, Ao, Ti, Ai}
    # Iterate over all dimension groups
    for (d, blocks_in) in in.dim_blocks
        if !haskey(out.dim_blocks, d)
            error("Output ASU missing dimension $d blocks")
        end
        
        blocks_out = out.dim_blocks[d]
        plans = plan.block_plans[d]
        
        Threads.@threads for i in 1:length(blocks_in)
            p = plans[i]
            b_in = blocks_in[i]
            b_out = blocks_out[i]
            
            # Forward: mul!(b_out.data, p, b_in.data)
            mul!(b_out.data, p, b_in.data)
        end
    end
    return out
end

"""
    ldiv!(out::CrystallographicASU, plan::CFFTPlan, in::CrystallographicASU)

Perform the inverse Crystallographic FFT (Spectral -> Spatial).
"""
function ldiv!(out::CrystallographicASU{D, To, Ao}, plan::CFFTPlan, in::CrystallographicASU{D, Ti, Ai}) where {D, To, Ao, Ti, Ai}
    for (d, blocks_in) in in.dim_blocks
        if !haskey(out.dim_blocks, d)
            error("Output ASU missing dimension $d blocks")
        end
        
        blocks_out = out.dim_blocks[d]
        plans = plan.block_plans[d]
        
        if length(blocks_in) != length(blocks_out) || length(blocks_in) != length(plans)
             error("Mismatched block counts in dimension $d")
        end
        
        Threads.@threads for i in 1:length(blocks_in)
            p = plans[i]
            b_in = blocks_in[i]
            b_out = blocks_out[i]
            
            # Inverse: ldiv!(b_out.data, p, b_in.data)
            ldiv!(b_out.data, p, b_in.data)
        end
    end
    return out
end
