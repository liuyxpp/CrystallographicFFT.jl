module QFusedKRFFT

using LinearAlgebra
using LinearAlgebra: LAPACK
using FFTW
using ..SymmetryOps: SymOp, get_ops, check_shift_invariance, dual_ops,
    detect_centering_type, CentP
using ..ASU: find_optimal_shift
using ..SpectralIndexing: calc_spectral_asu, SpectralIndexing
using ..KRFFT: auto_L, _select_rep_ops, SubgridCenteringFoldPlan,
    plan_centering_fold, centering_fold!, fft_channels!, assemble_G0!,
    ifft_channels!, centering_unfold!, disassemble_G0!,
    GeneralCFFTPlan, plan_krfft, fft_reconstruct!, fast_reconstruct!,
    M2BackwardPlan, plan_m2_backward, execute_m2_backward!

export M2QPlan, plan_m2_q, execute_m2_q!
export M7SCFTPlan, plan_m7_scft, execute_m7_scft!
export M2SCFTPlan, plan_m2_scft, execute_m2_scft!, update_m2_kernel!
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
    rep_ops::Vector{<:SymOp}
    L::NTuple{D,Int}
    M::NTuple{D,Int}
    N::NTuple{D,Int}
    fill_map::Array{Int32}  # N1×N2×N3, maps full-grid to subgrid linear index
    # --- Pmmm separable butterfly fast path ---
    is_separable::Bool                    # true if Pmmm-like (diagonal R, L=[2,2,2])
    K_fiber::Array{Float64, 4}            # (d, M1, M2, M3) — real K values per fiber
    twiddle_1d::Vector{Vector{ComplexF64}} # twiddle_1d[dim][q_d+1] = exp(2πi q_d/N_d)
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

    # 3. Select one representative operation per subgrid (shared logic)
    rep_ops = _select_rep_ops(shifted_ops, L, collect(N), D)
    d = prod(L)

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

    # 8. Detect Pmmm-like separable structure and precompute fast-path data
    is_sep = _is_pmmm_like(rep_ops, L, D, N)
    K_fiber = zeros(Float64, 0, 0, 0, 0)  # placeholder
    tw_1d = Vector{ComplexF64}[]
    if is_sep
        K_fiber, tw_1d = _build_separable_data(rep_ops, L, M_sub, collect(N), D,
                                                kernel_func)
    end

    return M2QPlan{D, typeof(sub_fft_plan), typeof(sub_ifft_plan)}(
                      Q_first_row, gather_idx,
                      sub_fft_plan, sub_ifft_plan,
                      Y_buf, Y_new_buf,
                      rep_ops, Tuple(L), Tuple(M_sub), N, fill_map,
                      is_sep, K_fiber, tw_1d)
end

# ============================================================================
# Q Matrix Construction (internal)
# ============================================================================

"""
    _build_q_matrices!(Q_first_row, gather_idx, rep_ops, L, M_sub, N, D, kernel_func)

Build Q = B⁻¹·diag(K)·B matrices for all fibers, store first row only.
"""
function _build_q_matrices!(Q_first_row, gather_idx,
                            rep_ops, L, M_sub, N, D, kernel_func)
    d = length(rep_ops)

    # Extract α for each rep_op as NTuple (canonical ordering)
    alphas = Vector{NTuple{3,Int}}(undef, d)
    idx = 0
    for x0 in Iterators.product([0:L[dd]-1 for dd in 1:D]...)
        idx += 1
        alphas[idx] = (x0[1], D >= 2 ? x0[2] : 0, D >= 3 ? x0[3] : 0)
    end

    # Pre-extract R matrices and translations from SymOps
    R_flat = Array{Int}(undef, D, D, d)
    t_flat = Vector{NTuple{3,Int}}(undef, d)
    for a in 1:d
        g = rep_ops[a]
        t_int = round.(Int, g.t)
        t_flat[a] = (t_int[1], D >= 2 ? t_int[2] : 0, D >= 3 ? t_int[3] : 0)
        for i in 1:D, j in 1:D
            R_flat[i, j, a] = round(Int, g.R[i, j])
        end
    end

    # Pre-allocate all working buffers outside the loop
    B_matrix = zeros(ComplexF64, d, d)
    B_copy = zeros(ComplexF64, d, d)
    KB = zeros(ComplexF64, d, d)
    K_values = zeros(Float64, d)
    h_vec = zeros(Int, D)
    h_centered = zeros(Int, D)
    rot_h = zeros(Int, D)
    q_vec = zeros(Int, D)
    ipiv = zeros(LinearAlgebra.BlasInt, d)

    N1 = N[1]; N2 = D >= 2 ? N[2] : 1; N3 = D >= 3 ? N[3] : 1
    M1 = M_sub[1]; M2v = D >= 2 ? M_sub[2] : 1; M3 = D >= 3 ? M_sub[3] : 1

    for q_cart in CartesianIndices(Tuple(M_sub))
        for dd in 1:D
            q_vec[dd] = q_cart[dd] - 1
        end

        # --- Build B matrix and K values (single clean pass) ---
        for a in 1:d
            α = alphas[a]

            # Full-grid frequency h_a = q + M * α_a
            h_vec[1] = q_vec[1] + M1 * α[1]
            if D >= 2; h_vec[2] = q_vec[2] + M2v * α[2]; end
            if D >= 3; h_vec[3] = q_vec[3] + M3 * α[3]; end

            # Centered frequency for kernel
            h_centered[1] = h_vec[1] >= N1 ÷ 2 ? h_vec[1] - N1 : h_vec[1]
            if D >= 2; h_centered[2] = h_vec[2] >= N2 ÷ 2 ? h_vec[2] - N2 : h_vec[2]; end
            if D >= 3; h_centered[3] = h_vec[3] >= N3 ÷ 2 ? h_vec[3] - N3 : h_vec[3]; end
            K_values[a] = kernel_func(h_centered)

            # B[a, b] = exp(-2πi h_a · t_b / N)
            for b in 1:d
                tb = t_flat[b]
                phase_val = h_vec[1] * tb[1] / N1
                if D >= 2; phase_val += h_vec[2] * tb[2] / N2; end
                if D >= 3; phase_val += h_vec[3] * tb[3] / N3; end
                @inbounds B_matrix[a, b] = exp(-im * 2π * phase_val)
            end
        end

        # --- Gather indices: R_b^T q mod M (independent of α) ---
        for b in 1:d
            for d1 in 1:D
                s = 0
                for d2 in 1:D
                    @inbounds s += R_flat[d2, d1, b] * q_vec[d2]
                end
                rot_h[d1] = mod(s, M_sub[d1])
            end
            lin_rot = 1 + rot_h[1]
            if D >= 2; lin_rot += rot_h[2] * M1; end
            if D >= 3; lin_rot += rot_h[3] * M1 * M2v; end
            gather_idx[b, q_cart] = Int32(lin_rot)
        end

        # --- Q = B⁻¹ · diag(K) · B, store first row ---
        # KB = diag(K) * B  (in-place, no allocation)
        for b in 1:d
            for a in 1:d
                @inbounds KB[a, b] = K_values[a] * B_matrix[a, b]
            end
        end
        # Solve B · X = KB  ⟹  X = B⁻¹ · KB = Q
        copyto!(B_copy, B_matrix)
        LAPACK.getrf!(B_copy, ipiv)
        LAPACK.getrs!('N', B_copy, ipiv, KB)  # KB ← Q

        # Store first row of Q
        for b in 1:d
            @inbounds Q_first_row[b, q_cart] = KB[1, b]
        end
    end
end

# ============================================================================
# Separable Detection & Precomputation
# ============================================================================

"""
Check if the representative operations form a Pmmm-like structure:
  - All rotation matrices are diagonal (mirrors/inversions only)
  - L = [2, 2, ...] in all dimensions
  - All translations are simple: t[d] mod N[d] ∈ {0, N[d]-1}
    (i.e., no fractional shifts like N/4 from screw axes or glide planes)

Non-symmorphic groups (Fd-3m, Ia-3d, Fddd, Cmcm, Ibam, Imma, etc.) may have
diagonal rotations but their glide/screw translations make the B matrix twiddle
factors incompatible with the separable WHT butterfly fast path. These groups
must use the generic Q-multiply path, which achieves machine precision.
"""
function _is_pmmm_like(rep_ops::Vector{<:SymOp}, L::Vector{Int}, D::Int,
                       N::Union{Tuple,Vector}=())
    # L must be [2, 2, ..., 2]
    all(l == 2 for l in L) || return false
    # All R must be diagonal
    for op in rep_ops
        for i in 1:D, j in 1:D
            i != j && op.R[i, j] != 0 && return false
        end
    end
    # All translations must be "simple" (0 or N-1 mod N, i.e., effectively 0 or -1)
    # This rejects non-symmorphic operations with fractional translations
    if !isempty(N)
        for op in rep_ops
            t = round.(Int, op.t)
            for dd in 1:D
                t_mod = mod(t[dd], Int(N[dd]))
                if t_mod != 0 && t_mod != Int(N[dd]) - 1
                    return false
                end
            end
        end
    end
    return true
end

"""
Precompute K_fiber values and 1D twiddle factors for the separable fast path.

For Pmmm-like groups with L=[2,2,2], d=8, the B matrix factors as:
    B = WHT₈ · diag(twiddle)
where WHT₈ = H⊗H⊗H (Hadamard), and twiddle depends only on q.

Returns:
  - `K_fiber`: (d, M₁, M₂, M₃) array of real kernel values K(h_a)
  - `twiddle_1d`: 1D twiddle arrays, twiddle_1d[dim][q+1] = exp(2πi q/N_d)
"""
function _build_separable_data(rep_ops, L, M_sub, N, D, kernel_func)
    d = prod(L)

    # Build alphas as NTuples (canonical ordering matching rep_ops)
    alphas = Vector{NTuple{3,Int}}(undef, d)
    idx = 0
    for x0 in Iterators.product([0:L[dd]-1 for dd in 1:D]...)
        idx += 1
        alphas[idx] = (x0[1], D >= 2 ? x0[2] : 0, D >= 3 ? x0[3] : 0)
    end

    # K_fiber[a, q1, q2, q3] = K(h_a) where h_a = q + M*α_a
    K_fiber = zeros(Float64, d, M_sub...)
    h_centered = zeros(Int, D)
    q_vec = zeros(Int, D)

    for q_cart in CartesianIndices(Tuple(M_sub))
        for dd in 1:D; q_vec[dd] = q_cart[dd] - 1; end
        for a in 1:d
            α = alphas[a]
            for dd in 1:D
                h = q_vec[dd] + M_sub[dd] * α[dd]
                h_centered[dd] = h >= N[dd] ÷ 2 ? h - N[dd] : h
            end
            K_fiber[a, q_cart] = kernel_func(h_centered)
        end
    end

    # 1D twiddle factors: twiddle_1d[dim][q+1] = exp(2πi q / N_d)
    twiddle_1d = [zeros(ComplexF64, M_sub[dd]) for dd in 1:D]
    for dd in 1:D
        for q in 0:M_sub[dd]-1
            twiddle_1d[dd][q+1] = exp(im * 2π * q / N[dd])
        end
    end

    return K_fiber, twiddle_1d
end

# ============================================================================
# Separable WHT Butterfly (8-point, in-place on length-8 buffer)
# ============================================================================

"""
Apply the 8-point Walsh-Hadamard Transform (WHT₈ = H⊗H⊗H) in-place.
H = [1 1; 1 -1]. The buffer `z` has length 8 indexed as (α₁, α₂, α₃)
in canonical order: (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1).

Each stage applies H along one dimension:
  Stage d: for each pair (i, i+stride), compute [a+b, a-b] where stride = 2^(d-1)
"""
@inline function wht8!(z::NTuple{8, ComplexF64})
    # Stage 1: dim 1 (stride=1, pairs: (1,2),(3,4),(5,6),(7,8))
    a1 = z[1] + z[2]; b1 = z[1] - z[2]
    a2 = z[3] + z[4]; b2 = z[3] - z[4]
    a3 = z[5] + z[6]; b3 = z[5] - z[6]
    a4 = z[7] + z[8]; b4 = z[7] - z[8]

    # Stage 2: dim 2 (stride=2, pairs: (1,3),(2,4),(5,7),(6,8))
    c1 = a1 + a2; d1 = a1 - a2
    c2 = b1 + b2; d2 = b1 - b2
    c3 = a3 + a4; d3 = a3 - a4
    c4 = b3 + b4; d4 = b3 - b4

    # Stage 3: dim 3 (stride=4, pairs: (1,5),(2,6),(3,7),(4,8))
    return (c1 + c3, c2 + c4, d1 + d3, d2 + d4,
            c1 - c3, c2 - c4, d1 - d3, d2 - d4)
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

When `plan.is_separable == true` (Pmmm-like groups), uses the WHT
butterfly fast path instead of dense Q-row multiplication.
"""
function execute_m2_q!(plan::M2QPlan, f0::Array{Float64})
    M = plan.M
    Y = plan.Y_buf
    Y_new = plan.Y_new_buf

    # Step 0: Copy real subgrid data to complex buffer
    @. Y = complex(f0)

    # Step 1: Forward FFT (in-place)
    plan.sub_fft_plan * Y

    # Step 2: Q multiply — dispatch to fast or generic path
    if plan.is_separable
        _q_multiply_separable!(Y_new, Y, plan)
    else
        _q_multiply_generic!(Y_new, Y, plan)
    end

    # Step 3: Inverse FFT (in-place, includes 1/prod(M) scaling)
    plan.sub_ifft_plan * Y_new

    # Step 4: Write real part back to f0
    @. f0 = real(Y_new)
end

"""
Generic Q multiplication: Y_new[q] = Σ_m Q[m,q] · Y[gather[m,q]]
"""
function _q_multiply_generic!(Y_new, Y, plan)
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
end

"""
Pmmm separable butterfly Q multiplication.

Algorithm for each q:
  1. Gather 8 values from Y using gather_idx
  2. Apply twiddle factors: z[m] *= tw₁(q₁)^α₁ · tw₂(q₂)^α₂ · tw₃(q₃)^α₃
  3. Forward WHT₈: w = WHT₈ · z  (3-stage butterfly, additions only)
  4. Multiply by K: v[m] = K_fiber[m,q] · w[m]  (8 real multiplies)
  5. Sum and normalize: Y_new[q] = (1/d) Σ v[m]

This replaces 8 complex Q-row multiplications with WHT butterfly + 8 real K multiplies.
"""
function _q_multiply_separable!(Y_new, Y, plan)
    idx = plan.gather_idx
    K = plan.K_fiber
    tw = plan.twiddle_1d
    M = plan.M
    d = length(plan.rep_ops)
    inv_d = 1.0 / d

    @inbounds for q3 in 1:M[3]
        tw3 = tw[3][q3]  # exp(2πi (q3-1)/N3)
        for q2 in 1:M[2]
            tw2 = tw[2][q2]
            tw23 = tw2 * tw3   # product of dim 2,3 twiddles (for α₂=α₃=1)
            for q1 in 1:M[1]
                tw1 = tw[1][q1]

                # Step 1: Gather 8 values from Y
                # Canonical alpha order: (0,0,0),(1,0,0),(0,1,0),(1,1,0),
                #                        (0,0,1),(1,0,1),(0,1,1),(1,1,1)
                y1 = Y[idx[1, q1, q2, q3]]  # α=(0,0,0), tw=1
                y2 = Y[idx[2, q1, q2, q3]]  # α=(1,0,0), tw=tw1
                y3 = Y[idx[3, q1, q2, q3]]  # α=(0,1,0), tw=tw2
                y4 = Y[idx[4, q1, q2, q3]]  # α=(1,1,0), tw=tw1*tw2
                y5 = Y[idx[5, q1, q2, q3]]  # α=(0,0,1), tw=tw3
                y6 = Y[idx[6, q1, q2, q3]]  # α=(1,0,1), tw=tw1*tw3
                y7 = Y[idx[7, q1, q2, q3]]  # α=(0,1,1), tw=tw2*tw3
                y8 = Y[idx[8, q1, q2, q3]]  # α=(1,1,1), tw=tw1*tw2*tw3

                # Step 2: Apply twiddle factors
                # twiddle[m] = prod_d tw_d^α_m_d
                tw12 = tw1 * tw2
                tw13 = tw1 * tw3
                tw123 = tw12 * tw3

                z = (y1,
                     y2 * tw1,
                     y3 * tw2,
                     y4 * tw12,
                     y5 * tw3,
                     y6 * tw13,
                     y7 * tw23,
                     y8 * tw123)

                # Step 3: Forward WHT₈
                w = wht8!(z)

                # Step 4: Multiply by K and sum
                acc = K[1, q1, q2, q3] * w[1] +
                      K[2, q1, q2, q3] * w[2] +
                      K[3, q1, q2, q3] * w[3] +
                      K[4, q1, q2, q3] * w[4] +
                      K[5, q1, q2, q3] * w[5] +
                      K[6, q1, q2, q3] * w[6] +
                      K[7, q1, q2, q3] * w[7] +
                      K[8, q1, q2, q3] * w[8]

                # Step 5: Normalize (WHT inverse first-row = [1/d, ...] )
                Y_new[q1, q2, q3] = acc * inv_d
            end
        end
    end
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

# ============================================================================
# M7 SCFT Plan: fold → FFT → assemble G₀ → Q-multiply → disassemble → IFFT → unfold
# ============================================================================

"""
    M7SCFTPlan

Plan for M7 SCFT diffusion step with centering fold acceleration.

Pipeline: f₀(M³) → centering_fold → n_ch × H³ → FFT → assemble G₀(M³)
          → Q-multiply G₀ → disassemble → n_ch × H³ → IFFT → unfold → f₀'(M³)

The centering fold reduces FFT/IFFT cost by 2× (I/C/A) or 4× (F) by decomposing
into smaller channel FFTs. The Q-multiply handles stride-2 aliasing on the M³ grid,
identical to M2's Q operation.

# Fields
- `m2q_plan`: The underlying M2QPlan (provides Q matrices, gather indices, buffers)
- `fold_plan`: SubgridCenteringFoldPlan for fold/unfold and channel FFTs
- `G0_view`: Reshaped view of m2q_plan.Y_buf for assemble/disassemble
- `M`: subgrid dimensions (M₁, M₂, M₃)
"""
struct M7SCFTPlan
    m2q_plan::M2QPlan
    fold_plan::SubgridCenteringFoldPlan
    G0_view::Array{ComplexF64,3}
    M::NTuple{3,Int}
end

"""
    plan_m7_scft(N, sg_num, dim, Δs, lattice) -> M7SCFTPlan

Create an M7 SCFT plan for the given space group and grid.
Combines centering fold (FFT acceleration) with M2's Q-multiply (aliasing).

Throws an error if the space group is P-centering (use M2+Q instead).
"""
function plan_m7_scft(N::NTuple{3,Int}, sg_num::Int, dim::Int,
                      Δs::Float64, lattice::AbstractMatrix)
    # First create the M2QPlan (handles Q matrices, buffers, etc.)
    m2q = plan_m2_q(N, sg_num, dim, Δs, lattice)

    M = NTuple{3,Int}(m2q.M)

    # Detect centering
    ops = get_ops(sg_num, dim, N)
    _, shifted_ops = find_optimal_shift(ops, N)
    cent = detect_centering_type(shifted_ops, N)
    if cent == CentP
        error("M7 SCFT requires centered lattice (I/F/C/A). " *
              "For P-centering, use M2+Q (plan_m2_q) instead.")
    end

    if !all(iseven, M)
        error("M sub-grid dimensions $M must be even for centering fold.")
    end

    # Create fold plan
    fold_plan = plan_centering_fold(cent, M)

    # Create G₀ view into M2Q's Y_buf
    G0_view = reshape(m2q.Y_buf, M)

    return M7SCFTPlan(m2q, fold_plan, G0_view, M)
end

"""
    execute_m7_scft!(plan::M7SCFTPlan, f0::Array{Float64,3})

Apply diffusion operator to subgrid data f₀ in-place using M7 SCFT pipeline.

Hot path:
  1. centering_fold: f₀(M³) → n_ch × H³
  2. fft_channels: n_ch × H³ → n_ch × Ĥ³
  3. assemble_G₀: n_ch × Ĥ³ → G₀(M³)     [write into Y_buf]
  4. Q-multiply: G₀ → G₀'                 [M2's Q on M³, result in Y_new_buf]
  5. copy: Y_new_buf → Y_buf (for disassemble)
  6. disassemble_G₀: G₀'(M³) → n_ch × Ĥ³
  7. ifft_channels: n_ch × Ĥ³ → n_ch × H³
  8. centering_unfold: n_ch × H³ → f₀'(M³)

Steps 1-3 replace M2's forward FFT (fewer FFT points via centering fold).
Step 4 is identical to M2's Q-multiply.
Steps 5-8 replace M2's inverse FFT.
"""
function execute_m7_scft!(plan::M7SCFTPlan, f0::Array{Float64,3})
    m2q = plan.m2q_plan
    fold = plan.fold_plan

    # Steps 1-2: Centering fold + channel FFTs
    centering_fold!(fold, f0)
    fft_channels!(fold)

    # Step 3: Assemble into M2Q's Y_buf (viewed as M³)
    assemble_G0!(plan.G0_view, fold)

    # Step 4: Q multiply (operates on M³ Y_buf → Y_new_buf)
    if m2q.is_separable
        _q_multiply_separable!(m2q.Y_new_buf, m2q.Y_buf, m2q)
    else
        _q_multiply_generic!(m2q.Y_new_buf, m2q.Y_buf, m2q)
    end

    # Step 5: Copy result for disassembly
    copyto!(m2q.Y_buf, m2q.Y_new_buf)

    # Steps 6-8: Disassemble + channel IFFTs + centering unfold
    disassemble_G0!(fold, plan.G0_view)
    ifft_channels!(fold)
    centering_unfold!(fold, f0)
end

# ============================================================================
# M2 SCFT Plan: forward → spectral K → backward (all space groups)
# ============================================================================

"""
    M2SCFTPlan

SCFT diffusion plan using M2 forward + spectral K multiply + M2 backward.

Computes `f₀_new = IKRFFT_M2(K(h) · KRFFT_M2(f₀))` where:
- `KRFFT_M2` is the M2 forward transform (subgrid FFT + fast_reconstruct)
- `K(h) = exp(-Δs · |k(h)|²)` is the diffusion kernel
- `IKRFFT_M2` is the M2 backward transform (inv_reconstruct + IFFT)

This replaces M2+Q's Q-matrix approach (`B⁻¹·diag(K)·B`) with direct scalar
multiplication in spectral space, achieving machine precision for ALL space
groups (including P-centering).

# Fields
- `fwd_plan`: M2 forward plan (`GeneralCFFTPlan`)
- `bwd_plan`: M2 backward plan (`M2BackwardPlan`)
- `K_spec`: Pre-computed diffusion kernel for each spectral ASU point
- `F_spec`: Workspace for spectral coefficients
- `n_spec`: Number of spectral ASU points
"""
struct M2SCFTPlan{FP<:GeneralCFFTPlan, BP<:M2BackwardPlan}
    fwd_plan::FP
    bwd_plan::BP
    K_spec::Vector{Float64}
    F_spec::Vector{ComplexF64}
    n_spec::Int
end

"""
    plan_m2_scft(N, sg_num, dim, Δs, lattice) -> M2SCFTPlan

Create an SCFT diffusion plan using M2 fwd+bwd for ANY space group.

# Arguments
- `N`: Full grid dimensions, e.g. `(64, 64, 64)`
- `sg_num`: Space group number (1–230)
- `dim`: Spatial dimension (2 or 3)
- `Δs`: Chain contour step size
- `lattice`: Lattice vectors as columns of a matrix (a₁|a₂|a₃)

# Notes
Unlike `plan_m7_scft` (centered only) or `plan_m2_q` (has precision issues),
this plan works for ALL space groups with machine precision.

When `Δs` changes, call `update_m2_kernel!` to recompute K in O(n_spec) time.
"""
function plan_m2_scft(N::Tuple, sg_num::Int, dim::Int,
                       Δs::Float64, lattice::AbstractMatrix)
    D = length(N)
    @assert D == dim

    # 1. Get operations and apply magic shift
    direct_ops = get_ops(sg_num, dim, N)
    _, shifted_ops = find_optimal_shift(direct_ops, N)

    # 2. Compute spectral ASU
    spec_asu = calc_spectral_asu(shifted_ops, dim, N)
    n_spec = length(spec_asu.points)

    # 3. Build forward and backward plans
    fwd_plan = plan_krfft(spec_asu, shifted_ops)
    bwd_plan = plan_m2_backward(spec_asu, shifted_ops)

    # 4. Pre-compute diffusion kernel K(h) = exp(-Δs · |k(h)|²)
    recip_B = 2π * inv(Matrix(lattice))'
    K_spec = Vector{Float64}(undef, n_spec)
    for (i, pt) in enumerate(spec_asu.points)
        h_centered = [pt.idx[d] >= N[d] ÷ 2 ? pt.idx[d] - N[d] : pt.idx[d]
                      for d in 1:D]
        k_vec = recip_B * h_centered
        K_spec[i] = exp(-dot(k_vec, k_vec) * Δs)
    end

    F_spec = Vector{ComplexF64}(undef, n_spec)

    return M2SCFTPlan(fwd_plan, bwd_plan, K_spec, F_spec, n_spec)
end

"""
    execute_m2_scft!(plan::M2SCFTPlan, f0::Array{Float64,3})

Apply the SCFT diffusion operator to stride-L subgrid data `f₀` in-place.

Hot path: `f₀ → FFT(M³) → reconstruct → K·F̂ → inv_reconstruct → IFFT → f₀'`

Works for ALL space groups. Replaces M2+Q's Q-multiply with spectral K multiply.
"""
function execute_m2_scft!(plan::M2SCFTPlan, f0::Array{Float64,3})
    fwd = plan.fwd_plan
    bwd = plan.bwd_plan
    K = plan.K_spec
    n_spec = plan.n_spec
    M = fwd.subgrid_dims  # already NTuple — no conversion needed
    M_vol = prod(M)

    # Size check: f0 must be M³ = N ÷ L
    if size(f0) != M
        error("f0 size $(size(f0)) does not match plan subgrid dims $M. " *
              "Use L=$(fwd.L_factors[1]) stride subgrid extraction.")
    end

    # Step 1: Copy f₀ (real M³) into forward plan's input_buffer (complex flat)
    buf_in = fwd.input_buffer
    @inbounds @simd for i in 1:M_vol
        buf_in[i] = complex(f0[i])
    end

    # Step 2: Forward M2 KRFFT — FFT + reconstruct → F̂[n_spec] in output_buffer
    fft_reconstruct!(fwd)

    # Step 3: Spectral K multiply — F̂(h) *= K(h) — in-place on output_buffer
    buf_out = fwd.output_buffer
    @inbounds @simd for i in 1:n_spec
        buf_out[i] *= K[i]
    end

    # Step 4: Backward M2 — inv_reconstruct + IFFT → f₀'
    #         (reads directly from output_buffer, no F_spec copy needed)
    f0_buf = execute_m2_backward!(bwd, buf_out)

    # Step 5: Write real part back to f₀
    @inbounds @simd for i in 1:M_vol
        f0[i] = real(f0_buf[i])
    end
end

"""
    update_m2_kernel!(plan::M2SCFTPlan, N, Δs_new, lattice)

Update the diffusion kernel for a new `Δs` value without rebuilding the plan.
This is O(n_spec) — much faster than rebuilding Q matrices.

Requires the space group info to be embedded in the plan's forward plan.
"""
function update_m2_kernel!(plan::M2SCFTPlan,
                            N::Tuple, sg_num::Int, dim::Int,
                            Δs_new::Float64, lattice::AbstractMatrix)
    D = length(N)
    direct_ops = get_ops(sg_num, dim, N)
    _, shifted_ops = find_optimal_shift(direct_ops, N)
    spec_asu = calc_spectral_asu(shifted_ops, dim, N)

    recip_B = 2π * inv(Matrix(lattice))'
    K = plan.K_spec
    for (i, pt) in enumerate(spec_asu.points)
        h_centered = [pt.idx[d] >= N[d] ÷ 2 ? pt.idx[d] - N[d] : pt.idx[d]
                      for d in 1:D]
        k_vec = recip_B * h_centered
        K[i] = exp(-dot(k_vec, k_vec) * Δs_new)
    end
end

end  # module QFusedKRFFT
