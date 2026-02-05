
using FFTW

struct CFFTPlan{D, T, A}
    asu::CrystallographicASU{D, T, A}
    # Map shape -> Plan. Shared across blocks of same size.
    # The value is the FFTW plan object.
    fft_plans::Dict{Tuple, Any}
    # Map dimension -> Vector of Plan references for each block in that dimension
    block_plans::Dict{Int, Vector{Any}}
end


"""
    plan_cfft(N::Tuple, sg_num::Int, T::Type=Float64, ArrayType=Array; dim::Int=length(N))

Create a Crystallographic FFT Plan.
"""
function plan_cfft(N::Tuple, sg_num::Int, T::Type=Float64, ArrayType=Array; dim::Int=length(N))
    # 1. Get Symmetry Operations
    ops = get_ops(sg_num, dim, N)
    
    # 2. Calculate ASU Points
    points = calc_asu(N, ops)
    
    # 3. Pack into ASU Blocks
    c_asu = pack_asu(points, N, T, ArrayType) # returns CrystallographicASU{D, T, A_concrete}
    
    # 4. Create Sub-Plans
    # We want to create plans for unique shapes to save resources.
    
    # Collect all unique shapes
    shapes = Set{Tuple}()
    
    # Iterate through all blocks
    for (d, blocks) in c_asu.dim_blocks
        for b in blocks
            push!(shapes, size(b.data))
        end
    end
    
    # Create plans for each unique shape
    # We assume C2C transform for now as per design discussion.
    # Input T is usually Real (Float64).
    # Ideally, we map Real -> Complex.
    # But CFFT might involve Complex fields even in real space for some groups?
    # For standard MDE, q is real.
    # But sub-blocks might be complex?
    # Let's assume we are planning for Complex data to be safe and generic.
    
    fft_plans = Dict{Tuple, Any}()
    
    for sz in shapes
        # Create a dummy array for planning
        # If ArrayType is CuArray, we need to generate CuArray.
        # Construct Complex{T} dummy
        
        # Check if ArrayType is a constructor that takes dimensions?
        # Usually ArrayType is Array or CuArray.
        # Array{Complex{T}}(undef, sz)
        
        if ArrayType == Array
            # Ensure we create a Complex array for FFT planning
            if T <: Real
                 dummy = Array{Complex{T}}(undef, sz)
            else
                 dummy = Array{T}(undef, sz)
            end
        else
            if T <: Real
                 dummy = ArrayType{Complex{T}}(undef, sz)
            else
                 dummy = ArrayType{T}(undef, sz)
            end
        end
        
        # Plan FFT (in-line or out-of-place?)
        # Let's plan in-place to save memory?
        # Or out-of-place?
        # FFTW default is often out-of-place or requires flags.
        # Let's use standard plan_fft which is usually OOP.
        # But we want to reuse output buffers if possible.
        # For now, keep it simple: OOP plan.
        
        p = plan_fft(dummy)
        fft_plans[sz] = p
    end
    
    # 5. Map blocks to plans
    block_plans = Dict{Int, Vector{Any}}()
    
    for (d_key, blocks) in c_asu.dim_blocks
        plans_vec = []
        for b in blocks
            sz = size(b.data)
            push!(plans_vec, fft_plans[sz])
        end
        block_plans[d_key] = plans_vec
    end
    
    return CFFTPlan(c_asu, fft_plans, block_plans)
end
