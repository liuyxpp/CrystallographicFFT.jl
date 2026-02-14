module QFusedKRFFT

using LinearAlgebra
using FFTW
using ..SymmetryOps: SymOp, get_ops, check_shift_invariance, dual_ops
using ..ASU: find_optimal_shift
using ..KRFFT: auto_L

export M2QPlan, plan_m2_q, execute_m2_q!
export subgrid_to_fullgrid!, fullgrid_to_subgrid!

# ============================================================================
# Data Structures
# ============================================================================

"""
    M2QPlan{D}

Plan for the M2 Q-fused SCFT diffusion kernel.

The hot path (`execute_m2_q!`) operates entirely on the M-grid subgrid (ASU),
performing FFT → Q·Y → IFFT without any full-grid pack/symmetry fill.

# Fields
- `Q_first_row`: Pre-computed Q matrix first row, shape `(d, M₁, M₂, M₃)`
- `gather_idx`: Gather indices for rotated frequencies, shape `(d, M₁, M₂, M₃)`
- `sub_fft_plan`: FFTW plan for forward FFT on M-grid
- `sub_ifft_plan`: FFTW plan for inverse FFT on M-grid
- `Y_buf`, `Y_new_buf`: Work buffers of size `(M₁, M₂, M₃)`
- `rep_ops`: Representative symmetry operations (one per subgrid)
- `L`: Stride factors `[L₁, L₂, L₃]`
- `M`: Subgrid dimensions `[M₁, M₂, M₃]`
- `N`: Full grid dimensions `[N₁, N₂, N₃]`
- `fill_map`: Pre-computed map for `subgrid_to_fullgrid!`, shape `N₁×N₂×N₃`,
              each entry is the linear index into the subgrid `f₀`.
"""
struct M2QPlan{D, FP, IP}
    Q_first_row::Array{ComplexF64, 4}   # (d, M1, M2, M3)
    gather_idx::Array{Int32, 4}          # (d, M1, M2, M3)
    sub_fft_plan::FP
    sub_ifft_plan::IP
    Y_buf::Array{ComplexF64, D}
    Y_new_buf::Array{ComplexF64, D}
    rep_ops::Vector{SymOp}
    L::Vector{Int}
    M::Vector{Int}
    N::Vector{Int}
    fill_map::Array{Int32}  # N1×N2×N3, maps full-grid to subgrid linear index
end

# ============================================================================
# Planning
# ============================================================================

"""
    plan_m2_q(N::Tuple, sg_num::Int, dim::Int, Δs::Float64,
              lattice::AbstractMatrix; kwargs...) -> M2QPlan

Construct a Q-fused KRFFT plan for SCFT diffusion.

# Arguments
- `N`: Full grid dimensions, e.g. `(64, 64, 64)`
- `sg_num`: Space group number (1–230)
- `dim`: Spatial dimension (2 or 3)
- `Δs`: Chain contour step size
- `lattice`: Lattice vectors as columns of a matrix

# Returns
An `M2QPlan` ready for use with `execute_m2_q!`.
"""
function plan_m2_q(N::Tuple, sg_num::Int, dim::Int, Δs::Float64,
                   lattice::AbstractMatrix)
    D = length(N)
    @assert D == dim

    # 1. Get operations and apply magic shift
    direct_ops = get_ops(sg_num, dim, N)
    _, shifted_ops = find_optimal_shift(direct_ops, N)

    # 2. Auto-determine L factors
    L = auto_L(shifted_ops)
    M_sub = [N[d] ÷ L[d] for d in 1:D]

    if any(L .* M_sub .!= collect(N))
        error("Grid size N=$N not divisible by auto L=$L.")
    end

    # 3. Select one representative operation per subgrid
    subgrid_reps = Dict{Vector{Int}, SymOp}()
    subgrid_quality = Dict{Vector{Int}, Int}()

    for op in shifted_ops
        t = round.(Int, op.t)
        x0 = [mod(t[d], L[d]) for d in 1:D]
        is_diag = all(op.R[i,j] == 0 for i in 1:D for j in 1:D if i != j)
        simple_t = all(mod(t[d], N[d]) ∈ (0, N[d]-1) for d in 1:D)
        quality = is_diag ? (simple_t ? 2 : 1) : 0

        if !haskey(subgrid_reps, x0) || quality > subgrid_quality[x0]
            subgrid_reps[x0] = op
            subgrid_quality[x0] = quality
        end
    end

    # Enumerate subgrids in canonical order
    d = prod(L)  # number of subgrids = prod(L)
    rep_ops = Vector{SymOp}(undef, d)
    sub_idx = 0
    for x0 in Iterators.product([0:L[dd]-1 for dd in 1:D]...)
        sub_idx += 1
        x0_vec = collect(x0)
        if haskey(subgrid_reps, x0_vec)
            rep_ops[sub_idx] = subgrid_reps[x0_vec]
        else
            error("Subgrid x₀=$x0_vec not reachable. auto_L should have prevented this.")
        end
    end

    # 4. Build the diffusion kernel function
    recip_B = 2π * inv(lattice)'
    function kernel_func(h_vec::Vector{Int})
        K = recip_B * h_vec
        return exp(-dot(K, K) * Δs)
    end

    # 5. Build Q matrices and gather indices
    Q_first_row = zeros(ComplexF64, d, M_sub...)
    gather_idx = zeros(Int32, d, M_sub...)

    _build_q_matrices!(Q_first_row, gather_idx, rep_ops, L, M_sub, collect(N), D,
                       kernel_func)

    # 6. Plan FFTs on M-grid (in-place)
    Y_buf = zeros(ComplexF64, Tuple(M_sub))
    Y_new_buf = zeros(ComplexF64, Tuple(M_sub))
    sub_fft_plan = plan_fft!(Y_buf)
    sub_ifft_plan = plan_ifft!(Y_new_buf)

    # 7. Build fill_map for subgrid_to_fullgrid!
    fill_map = _build_fill_map(shifted_ops, L, M_sub, collect(N), D)

    return M2QPlan{D, typeof(sub_fft_plan), typeof(sub_ifft_plan)}(
                      Q_first_row, gather_idx,
                      sub_fft_plan, sub_ifft_plan,
                      Y_buf, Y_new_buf,
                      rep_ops, L, M_sub, collect(N), fill_map)
end

# ============================================================================
# Q Matrix Construction (internal)
# ============================================================================

"""
Build Q matrices for all fibers. For each subgrid frequency q ∈ [0,M)^D:
  1. Enumerate the d full-grid frequencies h that map to q via the rep_ops
  2. Build the d×d butterfly matrix B(q)
  3. Compute Q(q) = B⁻¹(q) · diag(K) · B(q)
  4. Store only Q's first row (since n_active=1 in M2)
"""
function _build_q_matrices!(Q_first_row, gather_idx,
                            rep_ops, L, M_sub, N, D, kernel_func)
    d = length(rep_ops)

    # Pre-compute what subgrid each rep_op maps q to
    # For rep_ops[a], the subgrid shift is x₀ = t_a mod L
    # The full-grid frequency is: for each combination of "high" bits α ∈ {0,1}^D,
    # h[dim] = q[dim] + M[dim] * α[dim]
    # But which α corresponds to which rep_op?

    # In plan_krfft's auto-L variant, rep_ops are enumerated in canonical order:
    # sub_idx = 1 corresponds to x₀ = (0,0,...,0)
    # sub_idx = 2 corresponds to x₀ = (1,0,...,0) (if L[1]=2)
    # etc.
    # The "subgrid parity" α for rep_ops[a] is reconstructed from the canonical order.

    # We need: for rep_ops[a] with parity α_a,
    #   h_a(q) = q + M .* α_a       (the full-grid frequency)
    #   rotated freq: R_a^T h_a mod M  (the gather source)
    #   weight: exp(-2πi h_a · t_a / N)

    # Extract α for each rep_op (canonical ordering matches Iterators.product)
    alphas = Vector{Vector{Int}}(undef, d)
    idx = 0
    for x0 in Iterators.product([0:L[dd]-1 for dd in 1:D]...)
        idx += 1
        alphas[idx] = collect(x0)
    end

    # Pre-allocate
    B_matrix = zeros(ComplexF64, d, d)
    K_values = zeros(Float64, d)
    h_vec = zeros(Int, D)
    rot_h = zeros(Int, D)

    for q_cart in CartesianIndices(Tuple(M_sub))
        q_vec = [q_cart[dd] - 1 for dd in 1:D]  # 0-based

        # Build B matrix and K values for this fiber
        fill!(B_matrix, zero(ComplexF64))

        for a in 1:d
            α = alphas[a]
            g = rep_ops[a]

            # Full-grid frequency
            for dd in 1:D
                h_vec[dd] = q_vec[dd] + M_sub[dd] * α[dd]
            end

            # Kernel value K(h) — need wrapped h for physical frequency
            # Convert h to centered representation for kernel
            h_centered = zeros(Int, D)
            for dd in 1:D
                h_centered[dd] = h_vec[dd] >= N[dd] ÷ 2 ? h_vec[dd] - N[dd] : h_vec[dd]
            end
            K_values[a] = kernel_func(h_centered)

            # Phase weight: exp(-2πi h · t_g / N)
            phase_val = 0.0
            for dd in 1:D
                phase_val += h_vec[dd] * g.t[dd] / N[dd]
            end
            weight = exp(-im * 2π * phase_val)

            # Rotated frequency: R_g^T h mod M
            for d1 in 1:D
                s = 0
                for d2 in 1:D
                    s += g.R[d2, d1] * h_vec[d2]  # R^T: swap d1,d2
                end
                rot_h[d1] = mod(s, M_sub[d1])
            end

            # Linear index of rotated freq in M-grid (1-based, column-major)
            lin_rot = 1
            stride = 1
            for dd in 1:D
                lin_rot += rot_h[dd] * stride
                stride *= M_sub[dd]
            end

            # B[a, col] = weight, where col is determined by rot_h's subgrid index
            # Since all d ops map to the SAME subgrid freq space (they differ by which
            # M-grid point they read from), col = the index among the d gather sources.
            # But actually B is the matrix mapping Y₀-values to F-values:
            # F(h_a) = Σ_b B[a,b] * Y₀(gather_src_b)
            # In the M2 case with n_active=1, Y₀ is a single M-grid, and all gather
            # sources are positions within that same M-grid.
            # The "column" is determined by which M-grid position is being read.
            # For the butterfly structure, col = a itself when referenced by the
            # canonical ordering of the ops.
            # Actually, let me reconsider: the B matrix relates:
            # F(h_a) = Σ_b w_b(h_a) · Y₀(R_b^T h_a mod M)
            # So B[a, b] = w_b(h_a) IF R_b^T h_a mod M = R_b^T h_a mod M
            # This is a d×d matrix indexed by (fiber member a, op b).

            # Store gather index for this (a, q)
            gather_idx[a, q_cart] = Int32(lin_rot)

            # B matrix: row = a (fiber member), col = op index
            # Each row has exactly d entries corresponding to the d ops
            # B[a, b] = exp(-2πi h_a · t_b / N) if R_b^T h_a mod M maps to the
            # correct subgrid frequency. But this can produce different target indices.
            # Let me just build the full B matrix properly.
        end

        # Build B properly: B[a, b] tells the contribution of Y₀(rot_b(h_a)) to F(h_a)
        # with weight w_b(h_a).
        # Note: For each (a, b), h_a is fixed (determined by α_a), and we apply op b.
        fill!(B_matrix, zero(ComplexF64))
        # We also need a gather map: for each (a, b), what M-grid linear index does
        # Y₀(R_b^T h_a mod M) correspond to?
        gather_for_B = zeros(Int, d, d)

        for a in 1:d
            α = alphas[a]
            for dd in 1:D
                h_vec[dd] = q_vec[dd] + M_sub[dd] * α[dd]
            end

            for b in 1:d
                g = rep_ops[b]

                # Phase: exp(-2πi h_a · t_b / N)
                phase_val = 0.0
                for dd in 1:D
                    phase_val += h_vec[dd] * g.t[dd] / N[dd]
                end
                weight = exp(-im * 2π * phase_val)

                # Rotated freq: R_b^T h_a mod M
                for d1 in 1:D
                    s = 0
                    for d2 in 1:D
                        s += g.R[d2, d1] * h_vec[d2]
                    end
                    rot_h[d1] = mod(s, M_sub[d1])
                end

                lin_rot = 1
                stride = 1
                for dd in 1:D
                    lin_rot += rot_h[dd] * stride
                    stride *= M_sub[dd]
                end

                B_matrix[a, b] = weight
                gather_for_B[a, b] = lin_rot
            end
        end

        # Now: F(h_a) = Σ_b B[a,b] · Y₀[gather_for_B[a,b]]
        # In the standard M2 case, all gather_for_B[a, b] for fixed b differ across a,
        # but for fixed a, different b read different positions.
        # Actually, the gather depends on BOTH a and b. But in the Q formula,
        # we want: Q such that Y₀_new = Q · y_gathered
        # where y_gathered[b] = Y₀[gather_for_B[*, b]] ... this needs more thought.

        # SIMPLIFICATION: In M2 (auto_L), the rep_ops are structured such that
        # for the identity op (sub_idx=1, α=(0,...,0)), R=I and t=0.
        # For the other ops, R is a rotation/reflection and t is a translation.
        # The gather pattern for a fixed q is:
        #   For op b: rot_q_b = R_b^T q mod M
        # This is INDEPENDENT of the fiber member a (since R^T acts only on q part,
        # and h = q + M*α ⟹ R^T h mod M = R^T q mod M).
        # This is the key insight! The gather depends ONLY on b, not on a.

        # So gather_for_B[a, b] = gather_for_B[1, b] for all a.
        # Verify this and store gather_idx from b only.

        # Recompute gather indices using only q (independent of α):
        for b in 1:d
            g = rep_ops[b]
            for d1 in 1:D
                s = 0
                for d2 in 1:D
                    s += g.R[d2, d1] * q_vec[d2]
                end
                rot_h[d1] = mod(s, M_sub[d1])
            end
            lin_rot = 1
            stride = 1
            for dd in 1:D
                lin_rot += rot_h[dd] * stride
                stride *= M_sub[dd]
            end
            gather_idx[b, q_cart] = Int32(lin_rot)
        end

        # Now B[a,b] = exp(-2πi h_a · t_b / N)
        # And F(h_a) = Σ_b B[a,b] · Y₀[gather_idx[b]]
        # K_values[a] = K(h_a)
        # Q = B⁻¹ · diag(K) · B
        # Store only Q's first row: Q[1, :] = (B⁻¹ · diag(K) · B)[1, :]

        # Rebuild B matrix cleanly since gather is independent of a:
        fill!(B_matrix, zero(ComplexF64))
        for a in 1:d
            α = alphas[a]
            for dd in 1:D
                h_vec[dd] = q_vec[dd] + M_sub[dd] * α[dd]
            end

            # K value
            h_centered = zeros(Int, D)
            for dd in 1:D
                h_centered[dd] = h_vec[dd] >= N[dd] ÷ 2 ? h_vec[dd] - N[dd] : h_vec[dd]
            end
            K_values[a] = kernel_func(h_centered)

            for b in 1:d
                g = rep_ops[b]
                phase_val = 0.0
                for dd in 1:D
                    phase_val += h_vec[dd] * g.t[dd] / N[dd]
                end
                B_matrix[a, b] = exp(-im * 2π * phase_val)
            end
        end

        # Q = B⁻¹ · diag(K) · B, take first row
        # Q_full = B \ (Diagonal(K_values) * B)  — more stable
        KB = Diagonal(K_values) * B_matrix
        Q_full = B_matrix \ KB

        # Store first row
        for b in 1:d
            Q_first_row[b, q_cart] = Q_full[1, b]
        end
    end
end

# ============================================================================
# Execution (Hot Path)
# ============================================================================

"""
    execute_m2_q!(plan::M2QPlan, f0::Array{Float64})

Apply the Q-fused diffusion operator to subgrid data `f₀` in-place.

This is the SCFT hot path: purely operates on the M-grid subgrid with
no full-grid allocation or symmetry fill.

Pipeline: f₀ → complex copy → FFT → Q·Y → IFFT → real part → f₀
"""
function execute_m2_q!(plan::M2QPlan, f0::Array{Float64})
    M = plan.M
    Y = plan.Y_buf
    Y_new = plan.Y_new_buf

    # Step 0: Copy real subgrid data to complex buffer
    @. Y = complex(f0)

    # Step 1: Forward FFT (in-place)
    plan.sub_fft_plan * Y

    # Step 2: Q multiply
    d = length(plan.rep_ops)
    Q = plan.Q_first_row
    idx = plan.gather_idx
    @inbounds for q_cart in CartesianIndices(Y)
        acc = zero(ComplexF64)
        for m in 1:d
            acc += Q[m, q_cart] * Y[idx[m, q_cart]]
        end
        Y_new[q_cart] = acc
    end

    # Step 3: Inverse FFT (in-place, includes 1/prod(M) scaling)
    plan.sub_ifft_plan * Y_new

    # Step 4: Write real part back to f0
    @. f0 = real(Y_new)
end

# ============================================================================
# Grid Conversion Utilities
# ============================================================================

"""
    subgrid_to_fullgrid!(f_full::Array{Float64}, f0::Array{Float64}, plan::M2QPlan)

Expand subgrid data f₀(M³) to full grid f(N³) using pre-computed fill_map.
Each full-grid point maps to exactly one subgrid point via symmetry.

Complexity: O(N³) — single pass gather.
"""
function subgrid_to_fullgrid!(f_full::Array{Float64}, f0::Array{Float64},
                              plan::M2QPlan)
    fill_map = plan.fill_map
    f0_vec = vec(f0)
    @inbounds for i in eachindex(f_full)
        f_full[i] = f0_vec[fill_map[i]]
    end
end

"""
    fullgrid_to_subgrid!(f0::Array{Float64}, f_full::Array{Float64}, plan::M2QPlan)

Extract stride-L subgrid f₀(M³) from full grid f(N³).
Equivalent to f₀[i,j,k] = f[(i-1)*L₁+1, (j-1)*L₂+1, (k-1)*L₃+1].

Complexity: O(M³)
"""
function fullgrid_to_subgrid!(f0::Array{Float64}, f_full::Array{Float64},
                              plan::M2QPlan)
    L = plan.L
    M = plan.M
    D = length(M)

    if D == 3
        @inbounds for k in 1:M[3], j in 1:M[2], i in 1:M[1]
            f0[i, j, k] = f_full[1+(i-1)*L[1], 1+(j-1)*L[2], 1+(k-1)*L[3]]
        end
    elseif D == 2
        @inbounds for j in 1:M[2], i in 1:M[1]
            f0[i, j] = f_full[1+(i-1)*L[1], 1+(j-1)*L[2]]
        end
    else
        # Generic fallback
        for ci in CartesianIndices(Tuple(M))
            src_idx = ntuple(d -> 1 + (ci[d]-1)*L[d], D)
            f0[ci] = f_full[src_idx...]
        end
    end
end

# ============================================================================
# Fill Map Construction (internal)
# ============================================================================

"""
Build a pre-computed map: for each full-grid point x ∈ [0,N)^D, find the
linear index into f₀ (the M-subgrid) such that f(x) = f₀[fill_map[x]].

For each x, we search through shifted_ops to find an op g such that
g(x) = R·x + t (mod N) lands on the stride-L subgrid (all components divisible by L).
Then fill_map[x] = linear_index(g(x) .÷ L) in the M-grid.
"""
function _build_fill_map(shifted_ops, L, M_sub, N, D)
    fill_map = zeros(Int32, Tuple(N))
    x = zeros(Int, D)
    x_rot = zeros(Int, D)

    for lin_idx in 1:prod(N)
        # Convert linear index to coordinate (0-based)
        rem = lin_idx - 1
        for d in 1:D
            x[d] = rem % N[d]
            rem = rem ÷ N[d]
        end

        found = false
        for op in shifted_ops
            # Apply op: x' = R·x + t (mod N)
            on_subgrid = true
            for d1 in 1:D
                s = round(Int, op.t[d1])
                for d2 in 1:D
                    s += op.R[d1, d2] * x[d2]
                end
                x_rot[d1] = mod(s, N[d1])
                if x_rot[d1] % L[d1] != 0
                    on_subgrid = false
                    break
                end
            end

            if on_subgrid
                # Compute linear index into M-grid (1-based, column-major)
                sub_lin = 1
                stride = 1
                for d in 1:D
                    sub_lin += (x_rot[d] ÷ L[d]) * stride
                    stride *= M_sub[d]
                end
                fill_map[lin_idx] = Int32(sub_lin)
                found = true
                break
            end
        end

        if !found
            error("No symmetry operation maps grid point $x to the subgrid. " *
                  "This should not happen if auto_L is correct.")
        end
    end

    return fill_map
end

end  # module QFusedKRFFT
