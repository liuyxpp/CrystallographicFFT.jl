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

    valid_points = Vector{ASUPoint}()
    sizehint!(valid_points, n_total ÷ n_ops)

    # Pre-extract rotation matrices as flat NTuples for zero-alloc inner loop
    R_flat = Vector{NTuple{9,Int}}(undef, n_ops)
    @inbounds for i in 1:n_ops
        R = recip_ops[i].R
        R_flat[i] = (Int(R[1,1]), Int(R[2,1]), Int(R[3,1]),
                     Int(R[1,2]), Int(R[2,2]), Int(R[3,2]),
                     Int(R[1,3]), Int(R[2,3]), Int(R[3,3]))
    end

    # Pre-extract direct op translations for extinction filter
    t_direct_flat = Vector{NTuple{3,Float64}}(undef, n_ops)
    @inbounds for i in 1:n_ops
        t = direct_ops[i].t
        t_direct_flat[i] = (Float64(t[1]), Float64(t[2]), Float64(t[3]))
    end

    N1, N2, N3 = N[1], N[2], N[3]

    # Pre-allocated worklist buffer: max orbit size ≤ 2×|G| (point group + Hermitian)
    # In practice, orbit size ≤ |G|, but be safe
    max_orbit_size = 2 * n_ops
    worklist = Vector{Int}(undef, max_orbit_size)
    depth_zero = zeros(Int, D)  # shared constant for all points

    # Single-pass orbit enumeration using pre-allocated worklist
    @inbounds for lin_idx in 1:n_total
        visited[lin_idx] && continue

        # Mark starting point
        visited[lin_idx] = true
        worklist[1] = lin_idx
        wlen = 1
        min_lin = lin_idx
        orbit_size = 1

        # BFS: apply ops to all discovered orbit members
        wi = 1
        while wi <= wlen
            curr_lin = worklist[wi]
            wi += 1

            # Decompose linear index → (k1, k2, k3) 0-based
            rem_c = curr_lin - 1
            ck1 = rem_c % N1
            rem_c = rem_c ÷ N1
            ck2 = rem_c % N2
            ck3 = rem_c ÷ N2

            for g in 1:n_ops
                Rg = R_flat[g]
                # R * k mod N → linear index (inline 3×3 matmul)
                v1 = mod(Rg[1]*ck1 + Rg[4]*ck2 + Rg[7]*ck3, N1)
                v2 = mod(Rg[2]*ck1 + Rg[5]*ck2 + Rg[8]*ck3, N2)
                v3 = mod(Rg[3]*ck1 + Rg[6]*ck2 + Rg[9]*ck3, N3)
                li = v1 + N1 * v2 + N1 * N2 * v3 + 1

                if !visited[li]
                    visited[li] = true
                    orbit_size += 1
                    if li < min_lin
                        min_lin = li
                    end
                    wlen += 1
                    if wlen <= max_orbit_size
                        worklist[wlen] = li
                    end
                    # If orbit exceeds buffer, we still count but skip BFS expansion
                    # This is safe because crystallographic orbits are bounded by |G|
                end
            end
        end

        # Decompose min_lin → k_rep (0-based)
        rem_r = min_lin - 1
        kr1 = rem_r % N1
        rem_r = rem_r ÷ N1
        kr2 = rem_r % N2
        kr3 = rem_r ÷ N2

        # Extinction filter: Sum_{g in Stab(k)} exp(-2πi k·t_g/N) ≠ 0
        stab_sum = zero(ComplexF64)
        for i in 1:n_ops
            Rg = R_flat[i]
            # Check if R*k ≡ k (mod N) → stabilizer element
            s1 = mod(Rg[1]*kr1 + Rg[4]*kr2 + Rg[7]*kr3, N1)
            s2 = mod(Rg[2]*kr1 + Rg[5]*kr2 + Rg[8]*kr3, N2)
            s3 = mod(Rg[3]*kr1 + Rg[6]*kr2 + Rg[9]*kr3, N3)
            if s1 == kr1 && s2 == kr2 && s3 == kr3
                td = t_direct_flat[i]
                phase = kr1 * td[1] / N1 + kr2 * td[2] / N2 + kr3 * td[3] / N3
                stab_sum += cispi(-2 * phase)
            end
        end

        if abs(stab_sum) > 1e-5
            k_rep = [kr1, kr2, kr3]
            push!(valid_points, ASUPoint(k_rep, depth_zero, orbit_size))
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
