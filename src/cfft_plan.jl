using FFTW
using LinearAlgebra

struct CFFTPlan{D, T, A}
    asu::CrystallographicASU{D, T, A}
    fft_plans::Dict{Tuple, Any}
    block_plans::Dict{Int, Vector{Any}}
end

"""
    plan_cfft(N::Tuple, sg_num::Int, T::Type=Float64, ArrayType=Array; dim::Int=length(N))

Create a Crystallographic FFT Plan using Magic Shift logic.
"""
function plan_cfft(N::Tuple, sg_num::Int, T::Type=Float64, ArrayType=Array; dim::Int=length(N))
    # 1. ASU Construction (Logic + Magic Shift Search)
    points, shift = calc_asu(sg_num, dim, N)
    
    # 2. Packing into Blocks
    c_asu = pack_asu(points, N, T, ArrayType; shift=shift)

    # 3. FFT Planning for Unique Shapes
    fft_plans = Dict{Tuple, Any}()
    unique_shapes = Set(size(b.data) for (_, blocks) in c_asu.dim_blocks for b in blocks)
    
    for sz in unique_shapes
        # Always plan for Complex{T} to support general phase factors
        T_plan = (T <: Complex) ? T : Complex{T}
        dummy = (ArrayType == Array) ? Array{T_plan}(undef, sz) : ArrayType{T_plan}(undef, sz)
        fft_plans[sz] = plan_fft(dummy) # Out-of-place default
    end
    
    # 4. Map Blocks to Plans
    block_plans = Dict{Int, Vector{Any}}()
    for (d, blocks) in c_asu.dim_blocks
        block_plans[d] = [fft_plans[size(b.data)] for b in blocks]
    end
    
    return CFFTPlan(c_asu, fft_plans, block_plans)
end

# Forward Transform (CFFT)
function LinearAlgebra.mul!(y::CrystallographicASU, p::CFFTPlan, x::CrystallographicASU)
    for (d, blocks) in x.dim_blocks
        plans = p.block_plans[d]
        y_blocks = y.dim_blocks[d]
        for i in eachindex(blocks)
            # y = P * x (FFT)
            mul!(y_blocks[i].data, plans[i], blocks[i].data)
        end
    end
    return y
end

# Inverse Transform (ICFFT)
function LinearAlgebra.ldiv!(y::CrystallographicASU, p::CFFTPlan, x::CrystallographicASU)
    for (d, blocks) in x.dim_blocks
        plans = p.block_plans[d]
        y_blocks = y.dim_blocks[d]
        for i in eachindex(blocks)
            # y = P \ x (IFFT)
            ldiv!(y_blocks[i].data, plans[i], blocks[i].data)
        end
    end
    return y
end
