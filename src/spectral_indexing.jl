module SpectralIndexing

using ..SymmetryOps
using ..ASU
using LinearAlgebra

export SpectralIndexing, calc_spectral_asu, get_k_vector

struct SpectralIndexing
    points::Vector{ASUPoint}
    ops::Vector{<:SymOp}
    N::Tuple
end


"""
    calc_spectral_asu(ops::Vector{<:SymOp}, dim::Int, N::Tuple) -> SpectralIndexing

Calculate the spectral ASU using direct orbit enumeration.
Bypasses recursive calc_asu for O(N³ × |G|) performance.
"""
function calc_spectral_asu(direct_ops::Vector{<:SymOp}, dim::Int, N::Tuple)
    D = length(N)
    N_vec = collect(N)
    n_total = prod(N)
    
    # Reciprocal-space point group: R* = (R⁻¹)ᵀ, t=0
    recip_ops = dual_ops(direct_ops)
    n_ops = length(recip_ops)
    
    # Visited mask: linear index → already assigned to an orbit
    visited = falses(n_total)
    
    # Pre-allocate buffers
    k = zeros(Int, D)
    k_rot = zeros(Int, D)
    
    valid_points = Vector{ASUPoint}()
    
    # Pre-extract rotation matrices for fast access
    R_mats = [op.R for op in recip_ops]
    
    # Iterate all k-points in lexicographic order
    for lin_idx in 1:n_total
        visited[lin_idx] && continue
        
        # Convert linear index to k-vector (0-based, column-major)
        rem = lin_idx - 1
        @inbounds for d in 1:D
            k[d] = rem % N[d]
            rem = rem ÷ N[d]
        end
        
        # Compute orbit
        orbit_size = 0
        min_lin = lin_idx  # Track canonical rep (smallest linear index)
        
        # Use the point itself
        visited[lin_idx] = true
        orbit_size += 1
        
        # Apply all ops to find orbit members
        for g in 1:n_ops
            R = R_mats[g]
            # R * k mod N → linear index
            @inbounds begin
                li = 0
                stride = 1
                for d in 1:D
                    s = 0
                    for j in 1:D
                        s += R[d, j] * k[j]
                    end
                    k_rot[d] = mod(s, N[d])
                    li += k_rot[d] * stride
                    stride *= N[d]
                end
            end
            li += 1  # 1-based
            
            if !visited[li]
                visited[li] = true
                orbit_size += 1
                if li < min_lin
                    min_lin = li
                end
            end
        end
        
        # Also apply orbit closure (iterate existing orbit members through ops)
        # For crystallographic point groups, single pass is usually sufficient
        # but we need to be safe for higher-order groups.
        # Use worklist approach:
        worklist = [lin_idx]
        wi = 1
        while wi <= length(worklist)
            curr_lin = worklist[wi]
            wi += 1
            
            # Convert back to k-vector
            rem_c = curr_lin - 1
            @inbounds for d in 1:D
                k_rot[d] = rem_c % N[d]
                rem_c = rem_c ÷ N[d]
            end
            
            for g in 1:n_ops
                R = R_mats[g]
                @inbounds begin
                    li = 0
                    stride = 1
                    for d in 1:D
                        s = 0
                        for j in 1:D
                            s += R[d, j] * k_rot[j]
                        end
                        val = mod(s, N[d])
                        li += val * stride
                        stride *= N[d]
                    end
                end
                li += 1
                
                if !visited[li]
                    visited[li] = true
                    orbit_size += 1
                    push!(worklist, li)
                    if li < min_lin
                        min_lin = li
                    end
                end
            end
        end
        
        # Convert min_lin to k-vector for the representative
        rem_r = min_lin - 1
        k_rep = zeros(Int, D)
        @inbounds for d in 1:D
            k_rep[d] = rem_r % N[d]
            rem_r = rem_r ÷ N[d]
        end
        
        # Extinction filter: Sum_{g in Stab(k)} exp(-2πi k·t_g/N) ≠ 0
        stab_sum = zero(ComplexF64)
        for (i, op) in enumerate(recip_ops)
            R = op.R
            is_stab = true
            @inbounds for d in 1:D
                s = 0
                for j in 1:D
                    s += R[d, j] * k_rep[j]
                end
                if (s - k_rep[d]) % N_vec[d] != 0
                    is_stab = false
                    break
                end
            end
            if is_stab
                t_direct = direct_ops[i].t
                phase = 0.0
                @inbounds for d in 1:D
                    phase += k_rep[d] * t_direct[d] / N[d]
                end
                stab_sum += cispi(-2 * phase)
            end
        end
        
        if abs(stab_sum) > 1e-5
            depth = zeros(Int, D)  # Not needed for spectral ASU
            push!(valid_points, ASUPoint(k_rep, depth, orbit_size))
        end
    end
    
    sort!(valid_points, by=p -> p.idx)
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
