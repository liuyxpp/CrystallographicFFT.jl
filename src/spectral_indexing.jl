module SpectralIndexing

using ..SymmetryOps
using ..ASU
using LinearAlgebra

export SpectralIndexing, calc_spectral_asu, get_k_vector

struct SpectralIndexing
    points::Vector{ASUPoint}
    ops::Vector{SymOp}
    N::Tuple
end


"""
    calc_spectral_asu(ops::Vector{SymOp}, dim::Int, N::Tuple) -> SpectralIndexing

Calculate the spectral ASU for a given set of direct space operations.
Used when custom operations (e.g. shifted) are provided.
"""
function calc_spectral_asu(direct_ops::Vector{SymOp}, dim::Int, N::Tuple)
    # 2. Compute dual operations for reciprocal space
    # R* = (R^-1)^T, t* = 0 (BUT we need original t for filtering!)
    recip_ops = dual_ops(direct_ops)
    
    # 3. Calculate ASU on the reciprocal grid
    raw_points = calc_asu(N, recip_ops)
    
    # 4. Filter forbidden modes (Extinctions)
    # Condition: Sum_{g in Stab(k)} exp(-i k . t_g) != 0
    valid_points = Vector{ASUPoint}()
    
    for p in raw_points
        k_idx = p.idx
        
        # Identify Stabilizer
        stab_sum = 0.0 + 0.0im
        stab_count = 0
        
        for (i, op) in enumerate(recip_ops)
            # Check if op stabilizes k_idx (modulo N)
            k_new = op.R * k_idx
            
            # Check equality mod N
            diff = (k_new .- k_idx) .% collect(N)
            if all(d -> d == 0, diff)
                # In Stabilizer
                stab_count += 1
                
                # Add phase exp(-i k . t_direct)
                t_direct = direct_ops[i].t
                
                phase = 0.0
                for d in 1:length(N)
                    phase += k_idx[d] * t_direct[d] / N[d]
                end
                
                stab_sum += exp(-im * 2π * phase)
            end
        end
        
        # If sum is non-zero, keep point.
        # Use tolerance
        if abs(stab_sum) > 1e-5
            push!(valid_points, p)
        end
    end
    
    return SpectralIndexing(valid_points, recip_ops, N)
end

"""
    calc_spectral_asu(sg_num, dim, N::Tuple) -> SpectralIndexing

Calculate the spectral ASU for a given space group and grid size.
Wrapper collecting operations from `sg_num`.
"""
function calc_spectral_asu(sg_num::Int, dim, N::Tuple)
    direct_ops = get_ops(sg_num, dim, N)
    return calc_spectral_asu(direct_ops, dim, N)
end

"""
    get_k_vector(indexing::SpectralIndexing, idx::Int) -> Vector{Int}

Get the k-vector (frequency) for the i-th point in the spectral ASU.
Handles wrapping of indices to (-N/2, N/2].
"""
function get_k_vector(indexing::SpectralIndexing, idx::Int)
    if idx < 1 || idx > length(indexing.points)
        error("Index $idx out of bounds for SpectralIndexing with $(length(indexing.points)) points")
    end
    
    p = indexing.points[idx]
    raw_k = p.idx
    
    # Convert 0..N-1 to -N/2..N/2
    k_vec = zeros(Int, length(raw_k))
    for d in 1:length(raw_k)
        n = indexing.N[d]
        val = raw_k[d]
        if val >= n/2 + (n%2==0 ? 0 : 0.5) # Standard fftshift logic: N=8, 0..3->0..3, 4..7->-4..-1. 
            # Actually standard Julia fft freq correctness:
            # 0, 1, 2, 3, -4, -3, -2, -1 for N=8
            # 4 is the Nyquist freq, usually mapped to -4.
            # val >= (n ÷ 2) + 1 ? val - n : val  <-- NO
            # Correct: if val > n/2 -> val - n
            # For N=8: 4 > 4 is false. 4 -> 4 ? No, usually 4 maps to -4 or 4. 
            # FFTW/Julia convention: result of fft contains frequencies:
            # 0, 1, ..., N/2-1, -N/2, ..., -1] if N is even.
            # index 1..N
            # 0: 0
            # 1: 1
            # ...
            # N/2: -N/2  (index N/2+1)
            # ...
            val = val >= (n+1)÷2 + 1 ? val - n : val # This is approximate.
            
            # Let's use canonical logic:
            # freq = mod(val + n/2, n) - n/2 ? No
            
            if val >= n - n÷2 # e.g. 8-4=4. val>=4 -> val-8. 4->-4. 5->-3.
                 k_vec[d] = val - n
            else
                 k_vec[d] = val
            end
        else
             k_vec[d] = val
        end
    end
    return k_vec
end

end
