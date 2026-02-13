"""
    Staged KRFFT for Pm-3m (SG 221)
    
    Implements the full A8(1) + P3c + yx three-stage decomposition from KRFFT V.
    Target: |G|=48 fold speedup for cubic Pm-3m symmetry.
    
    Architecture:
        Full grid (2N)³ → A8 stride-2 → f₀₀₀ (N³)
        f₀₀₀ → P3c recursive C₃ decomposition → GP leaves (FFT) + SP recursion
        Butterfly reconstruction (P3c → A8) → spectral output
"""

using FFTW
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────

"""yx recursion tree node. Each node represents one level of diagonal mirror decomposition.
Splits stride-2 in x,y dimensions only (z untouched)."""
struct YxNode
    level::Int
    size::NTuple{3,Int}           # (Mx, My, Mz) at this level
    half_xy::NTuple{3,Int}        # (Mx÷2, My÷2, Mz)
    gp_buf::Array{ComplexF64,3}   # f₀₁ real-space GP data
    gp_fft::Array{ComplexF64,3}   # F₀₁ FFT output
    sp_bufs::Vector{Array{ComplexF64,3}}     # [f₀₀, f₁₁]
    fft_plan::FFTW.cFFTWPlan{ComplexF64, -1, false, 3, NTuple{3,Int}}
    fft_work::Array{ComplexF64,3}  # preallocated work buffer for in-place FFT
    children::Vector{Union{YxNode, Nothing}}  # [child_00, child_11]
end

"""P3c recursion tree node. Each node represents one level of C₃ decomposition."""
struct P3cNode
    level::Int
    size::NTuple{3,Int}           # (M,M,M) at this level
    half::NTuple{3,Int}           # (M÷2, M÷2, M÷2)
    gp_bufs::Vector{Array{ComplexF64,3}}     # [f_001, f_110], real-space GP data
    gp_ffts::Vector{Array{ComplexF64,3}}     # [F_001, F_110], FFT outputs
    gp_yx::Vector{Union{YxNode, Nothing}}    # yx trees for GP sub-subgrids
    sp_bufs::Vector{Array{ComplexF64,3}}     # [f_000, f_111], real-space SP data
    fft_plan::FFTW.cFFTWPlan{ComplexF64, -1, false, 3, NTuple{3,Int}}
    fft_work::Array{ComplexF64,3}  # preallocated work buffer for in-place FFT
    children::Vector{Union{P3cNode, Nothing}}  # [child_000, child_111]
end

"""Top-level staged KRFFT plan for Pm-3m."""
struct StagedPm3mPlan
    N_full::NTuple{3,Int}         # Full grid (2N, 2N, 2N)
    N_half::NTuple{3,Int}         # Half grid (N, N, N)
    a8_buf::Array{ComplexF64,3}   # f₀₀₀ real-space data (N³)
    a8_fft::Array{ComplexF64,3}   # F₀₀₀ frequency-domain (N³)
    p3c_root::P3cNode
end

# ─────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────

"""Build a yx recursion tree for diagonal mirror decomposition.
Splits stride-2 in x,y only. Input grid is (Mx, My, Mz)."""
function build_yx_tree(M::NTuple{3,Int}, level::Int=0; min_size::Int=2)
    M2 = (M[1] ÷ 2, M[2] ÷ 2, M[3])  # only split x,y
    gp_buf = zeros(ComplexF64, M2)
    gp_fft = zeros(ComplexF64, M2)
    sp_bufs = [zeros(ComplexF64, M2) for _ in 1:2]
    fft_plan = plan_fft(zeros(ComplexF64, M2), flags=FFTW.ESTIMATE)
    fft_work = zeros(ComplexF64, M2)
    
    if M2[1] >= min_size && M2[1] % 2 == 0
        children = Union{YxNode,Nothing}[
            build_yx_tree(M2, level+1; min_size),
            build_yx_tree(M2, level+1; min_size)]
    else
        children = Union{YxNode,Nothing}[nothing, nothing]
    end
    
    return YxNode(level, M, M2, gp_buf, gp_fft, sp_bufs, fft_plan, fft_work, children)
end

"""Build a P3c recursion tree for C₃ decomposition of an M³ grid.
`use_yx`: if true, apply yx diagonal mirror reduction to GP sub-subgrids.
  Default is false because FFTW is faster than yx forward+reconstruct at typical sizes."""
function build_p3c_tree(M::NTuple{3,Int}, level::Int=0; min_size::Int=2, use_yx::Bool=false)
    M2 = M .÷ 2
    gp_bufs = [zeros(ComplexF64, M2) for _ in 1:2]
    gp_ffts = [zeros(ComplexF64, M2) for _ in 1:2]
    sp_bufs = [zeros(ComplexF64, M2) for _ in 1:2]
    fft_plan = plan_fft(zeros(ComplexF64, M2), flags=FFTW.ESTIMATE)
    fft_work = zeros(ComplexF64, M2)
    
    # Build yx trees for GP sub-subgrids (only if enabled)
    if use_yx && M2[1] >= min_size && M2[1] % 2 == 0
        gp_yx = Union{YxNode,Nothing}[
            build_yx_tree(M2; min_size),
            build_yx_tree(M2; min_size)]
    else
        gp_yx = Union{YxNode,Nothing}[nothing, nothing]
    end
    
    if M2[1] >= min_size && M2[1] % 2 == 0
        children = Union{P3cNode,Nothing}[
            build_p3c_tree(M2, level+1; min_size, use_yx),
            build_p3c_tree(M2, level+1; min_size, use_yx)]
    else
        children = Union{P3cNode,Nothing}[nothing, nothing]
    end
    
    return P3cNode(level, M, M2, gp_bufs, gp_ffts, gp_yx, sp_bufs, fft_plan, fft_work, children)
end

"""Create a staged KRFFT plan for Pm-3m.
`use_yx`: if true, apply yx diagonal mirror reduction to GP leaves (default: false)."""
function plan_staged_pm3m(N_full::NTuple{3,Int}; use_yx::Bool=false)
    @assert all(n -> n % 2 == 0, N_full) "Grid dimensions must be even"
    @assert N_full[1] == N_full[2] == N_full[3] "Must be cubic"
    N_half = N_full .÷ 2
    a8_buf = zeros(ComplexF64, N_half)
    a8_fft = zeros(ComplexF64, N_half)
    p3c_root = build_p3c_tree(N_half; use_yx)
    return StagedPm3mPlan(N_full, N_half, a8_buf, a8_fft, p3c_root)
end

# ─────────────────────────────────────────────────────────────────────────
# Forward: extraction + FFT
# ─────────────────────────────────────────────────────────────────────────

"""Extract f₀₀₀(x₁) = u(2x₁) from the full (2N)³ grid."""
function pack_a8!(plan::StagedPm3mPlan, u::AbstractArray{<:Real,3})
    N = plan.N_half
    @inbounds for k in 1:N[3], j in 1:N[2], i in 1:N[1]
        plan.a8_buf[i, j, k] = u[2i-1, 2j-1, 2k-1]
    end
end

"""Extract 4 representative stride-2 sub-subgrids from parent data (M³).
SP: (0,0,0) and (1,1,1) — C₃-invariant, recurse further.
GP: (0,0,1) and (1,1,0) — C₃ orbit representatives, FFT directly.
"""
function extract_p3c_subgrids!(node::P3cNode, data::Array{ComplexF64,3})
    M = node.size; M2 = node.half
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        node.sp_bufs[1][i,j,k] = data[2i-1, 2j-1, 2k-1]               # (0,0,0)
        node.sp_bufs[2][i,j,k] = data[mod1(2i,M[1]), mod1(2j,M[2]), mod1(2k,M[3])]  # (1,1,1)
        node.gp_bufs[1][i,j,k] = data[2i-1, 2j-1, mod1(2k,M[3])]      # (0,0,1)
        node.gp_bufs[2][i,j,k] = data[mod1(2i,M[1]), mod1(2j,M[2]), 2k-1]  # (1,1,0)
    end
end

# ─────────────────────────────────────────────────────────────────────────
# yx Forward: extraction + FFT
# ─────────────────────────────────────────────────────────────────────────

"""Extract 3 yx sub-subgrids from parent data (Mx×My×Mz).
SP: (0,0) and (1,1) — yx-mirror invariant.
GP: (0,1) — representative; (1,0) derived via F₁₀(h₁)=F₀₁(h_y,h_x,h_z)."""
function extract_yx_subgrids!(node::YxNode, data::Array{ComplexF64,3})
    M = node.size; M2 = node.half_xy
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        node.sp_bufs[1][i,j,k] = data[2i-1, 2j-1, k]                    # (0,0)
        node.sp_bufs[2][i,j,k] = data[mod1(2i,M[1]), mod1(2j,M[2]), k]  # (1,1)
        node.gp_buf[i,j,k]     = data[2i-1, mod1(2j,M[2]), k]           # (0,1)
    end
end

"""Recursively execute yx decomposition: extract → FFT GP → recurse SP."""
function execute_yx_forward!(node::YxNode, data::Array{ComplexF64,3})
    extract_yx_subgrids!(node, data)
    
    # FFT the GP representative (0,1)
    mul!(node.gp_fft, node.fft_plan, node.gp_buf)
    
    # Recurse or leaf-FFT the 2 SP sub-subgrids
    for (idx, child) in enumerate(node.children)
        if child !== nothing
            execute_yx_forward!(child, node.sp_bufs[idx])
        else
            copyto!(node.fft_work, node.sp_bufs[idx])
            mul!(node.sp_bufs[idx], node.fft_plan, node.fft_work)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# yx Butterfly reconstruction
# ─────────────────────────────────────────────────────────────────────────

"""
Bottom-up butterfly reconstruction for yx diagonal mirror.

KRFFT V §3 eq (13):
  F(h) = F₀₀(h₁) + tw₁₁·F₁₁(h₁) + tw₁₀·F₀₁(h_y,h_x,h_z) + tw₀₁·F₀₁(h₁)
where tw_{pq} = exp(-2πi(p·h_x + q·h_y)/M_x)

GP relation: F₁₀(h₁) = F₀₁(h_y, h_x, h_z)
"""
function reconstruct_yx!(result::Array{ComplexF64,3}, node::YxNode)
    M = node.size; M2 = node.half_xy
    
    # Bottom-up: reconstruct SP children first
    sp_freq = node.sp_bufs
    for (idx, child) in enumerate(node.children)
        if child !== nothing
            reconstruct_yx!(sp_freq[idx], child)
        end
    end
    
    F00 = sp_freq[1]; F11 = sp_freq[2]
    F01 = node.gp_fft
    M2x = M2[1]; M2y = M2[2]; M2z = M2[3]
    
    # Build mirror-swapped copy: F01_swap[i,j,k] = F01[j,i,k] (= F₁₀)
    F10 = similar(F01)
    @inbounds for k in 1:M2z, j in 1:M2y, i in 1:M2x
        F10[i,j,k] = F01[j,i,k]  # R_α^T: (h_x,h_y,h_z) → (h_y,h_x,h_z)
    end
    
    # Precompute 1D twiddles for x,y (not z — z is not split)
    tw1x = [cispi(-2i / M[1]) for i in 0:M2x-1]
    tw1y = [cispi(-2j / M[2]) for j in 0:M2y-1]
    
    # Iterate by (h₀_x, h₀_y) quadrant: h_d = h₁_d + M2_d · h₀_d
    @inbounds for m0 in 0:1, n0 in 0:1
        sx = (-1)^n0; sy = (-1)^m0
        for k in 1:M2z
            for j in 1:M2y
                twy = tw1y[j] * sy
                for i in 1:M2x
                    twx = tw1x[i] * sx
                    val = F00[i,j,k] + twx*twy * F11[i,j,k] +
                          twx * F10[i,j,k] + twy * F01[i,j,k]
                    result[i + n0*M2x, j + m0*M2y, k] = val
                end
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# P3c Forward: extraction + FFT (with yx for GP leaves)
# ─────────────────────────────────────────────────────────────────────────

"""Recursively execute P3c decomposition: extract → yx+FFT GP → recurse SP."""
function execute_p3c_forward!(node::P3cNode, data::Array{ComplexF64,3})
    extract_p3c_subgrids!(node, data)
    
    # Process the 2 GP representatives through yx decomposition or plain FFT
    for idx in 1:2
        if node.gp_yx[idx] !== nothing
            # yx decomposition: recursively decompose, then reconstruct to get FFT output
            execute_yx_forward!(node.gp_yx[idx], node.gp_bufs[idx])
            reconstruct_yx!(node.gp_ffts[idx], node.gp_yx[idx])
        else
            mul!(node.gp_ffts[idx], node.fft_plan, node.gp_bufs[idx])
        end
    end
    
    # Recurse or leaf-FFT the 2 SP sub-subgrids
    for (idx, child) in enumerate(node.children)
        if child !== nothing
            execute_p3c_forward!(child, node.sp_bufs[idx])
        else
            copyto!(node.fft_work, node.sp_bufs[idx])
            mul!(node.sp_bufs[idx], node.fft_plan, node.fft_work)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Butterfly reconstruction: sub-FFTs → F₀₀₀(h)
# ─────────────────────────────────────────────────────────────────────────

"""
Bottom-up butterfly reconstruction of F on M³ from 4 sub-FFTs.

C₃ orbit relations (verified):
  Orbit 1: F_{001} → F_{100} → F_{010} → F_{001}  (via α: (n,m,l)→(m,l,n))
    F_{100}(h) = F_{001}(R_α^T h) = F_{001}(h₂,h₃,h₁)
    F_{010}(h) = F_{001}(R_{α²}^T h) = F_{001}(h₃,h₁,h₂)
    
  Orbit 2: F_{110} → F_{101} → F_{011} → F_{110}
    F_{101}(h) = F_{110}(R_{α²}^T h) = F_{110}(h₃,h₁,h₂)
    F_{011}(h) = F_{110}(R_α^T h)  = F_{110}(h₂,h₃,h₁)

Optimization: precomputes 1D twiddle vectors and C₃-rotated array copies
to avoid per-element cispi() calls. Iterates by h₀-octant for contiguous writes.
"""
function reconstruct_p3c!(result::Array{ComplexF64,3}, node::P3cNode)
    M = node.size; M2 = node.half
    
    # Bottom-up: reconstruct SP children first
    sp_freq = node.sp_bufs
    for (idx, child) in enumerate(node.children)
        if child !== nothing
            reconstruct_p3c!(sp_freq[idx], child)
        end
    end
    
    F000 = sp_freq[1]; F111 = sp_freq[2]
    F001 = node.gp_ffts[1]; F110 = node.gp_ffts[2]
    M2_1 = M2[1]; M2_2 = M2[2]; M2_3 = M2[3]
    
    # Build C₃-rotated copies of GP arrays (avoids per-element index permutation)
    F100 = similar(F001); F010 = similar(F001)
    F101 = similar(F110); F011 = similar(F110)
    @inbounds for k in 1:M2_3, j in 1:M2_2, i in 1:M2_1
        F100[i,j,k] = F001[j,k,i]  # R_α^T
        F010[i,j,k] = F001[k,i,j]  # R_{α²}^T
        F101[i,j,k] = F110[k,i,j]  # R_{α²}^T
        F011[i,j,k] = F110[j,k,i]  # R_α^T
    end
    
    # Precompute 1D twiddle vectors (valid for h₁ ∈ [0,M2))
    tw1x = [cispi(-2i / M[1]) for i in 0:M2_1-1]
    tw1y = [cispi(-2j / M[2]) for j in 0:M2_2-1]
    tw1z = [cispi(-2k / M[3]) for k in 0:M2_3-1]
    
    # Iterate by h₀ octant: h = h₁ + M2·h₀ where h₀ = (n₀,m₀,l₀) ∈ {0,1}³
    # phi_d(h) = tw1d[h₁+1] × (-1)^h₀d  (since cispi(-2(h₁+M2·h₀)/M) = tw1d × cispi(-h₀))
    @inbounds for l0 in 0:1, m0 in 0:1, n0 in 0:1
        sx = (-1)^n0; sy = (-1)^m0; sz = (-1)^l0
        for k in 1:M2_3
            twz = tw1z[k] * sz
            for j in 1:M2_2
                twy = tw1y[j] * sy
                twyz = twy * twz
                for i in 1:M2_1
                    twx = tw1x[i] * sx
                    val = F000[i,j,k] + 
                          twz * F001[i,j,k] + twy * F010[i,j,k] + twx * F100[i,j,k] +
                          twyz * F011[i,j,k] + twx*twz * F101[i,j,k] + twx*twy * F110[i,j,k] +
                          twx*twyz * F111[i,j,k]
                    result[i + n0*M2_1, j + m0*M2_2, k + l0*M2_3] = val
                end
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# S₃ symmetry utilities
# ─────────────────────────────────────────────────────────────────────────

"""Compute S₃ canonical representative of (i,j,k) mod M.
S₃ acts by permuting the 3 indices. Canonical = lexicographic minimum."""
function s3_canonical(i::Int, j::Int, k::Int, M::Int)
    a, b, c = mod(i, M), mod(j, M), mod(k, M)
    # Sort to get canonical representative
    if a > b; a, b = b, a; end
    if b > c; b, c = c, b; end
    if a > b; a, b = b, a; end
    return (a, b, c)
end

# ─────────────────────────────────────────────────────────────────────────
# Pruned butterfly reconstruction: only compute ASU-needed output points
# ─────────────────────────────────────────────────────────────────────────

"""Precomputed per-point data for pruned butterfly at a single P3c node."""
struct PrunedPoint
    i1p1::Int32   # h₁_x+1 (1-based index into sub-FFT)
    j1p1::Int32   # h₁_y+1
    k1p1::Int32   # h₁_z+1
    oi::Int32     # output index x (1-based)
    oj::Int32     # output index y
    ok::Int32     # output index z
    twx::ComplexF64  # precomputed twiddle × sign (includes (-1)^h₀)
    twy::ComplexF64
    twz::ComplexF64
end

"""S₃ fill pair: copy value from source index to target index in SP array."""
struct S3FillPair
    ti::Int32; tj::Int32; tk::Int32  # target (1-based)
    si::Int32; sj::Int32; sk::Int32  # source (1-based, S₃ canonical)
end

"""Precomputed pruned reconstruction plan for a single P3c node."""
struct PrecompPlan
    node_ref::P3cNode
    # Preallocated rotation buffers
    F100::Array{ComplexF64,3}
    F010::Array{ComplexF64,3}
    F101::Array{ComplexF64,3}
    F011::Array{ComplexF64,3}
    # Precomputed 1D twiddles (for full butterfly fallback)
    tw1x::Vector{ComplexF64}
    tw1y::Vector{ComplexF64}
    tw1z::Vector{ComplexF64}
    # Pruning control
    use_full::Bool
    pruned_pts::Vector{PrunedPoint}
    # S₃ fill pairs for SP children (applied after child butterfly)
    s3_fill::Vector{S3FillPair}
    # Children plans (for SP recursive nodes)
    children_plans::Vector{Union{PrecompPlan, Nothing}}
end

"""
    build_precomp_plan(node, needed) -> PrecompPlan

Build a precomputed pruned reconstruction plan for the P3c tree rooted at `node`.
`needed` is the set of output points (0-indexed tuples) that must be computed.
All twiddles, rotation buffers, and per-point data are precomputed for zero-allocation execution.
"""
function build_precomp_plan(node::P3cNode, needed::Vector{NTuple{3,Int}})
    M = node.size; M2 = node.half
    M2_1 = M2[1]; M2_2 = M2[2]; M2_3 = M2[3]
    
    # Preallocate rotation buffers
    F100 = similar(node.gp_ffts[1])
    F010 = similar(node.gp_ffts[1])
    F101 = similar(node.gp_ffts[2])
    F011 = similar(node.gp_ffts[2])
    
    # Precompute 1D twiddles
    tw1x = [cispi(-2i / M[1]) for i in 0:M2_1-1]
    tw1y = [cispi(-2j / M[2]) for j in 0:M2_2-1]
    tw1z = [cispi(-2k / M[3]) for k in 0:M2_3-1]
    
    # Decide full vs pruned (full butterfly is faster when most points are needed)
    use_full = length(needed) >= 0.8 * prod(M)
    
    # Precompute per-point data for pruned mode
    pruned_pts = PrunedPoint[]
    if !use_full
        for h in needed
            i1 = mod(h[1], M2_1); j1 = mod(h[2], M2_2); k1 = mod(h[3], M2_3)
            n0 = h[1] ÷ M2_1; m0 = h[2] ÷ M2_2; l0 = h[3] ÷ M2_3
            twx = tw1x[i1+1] * ((-1)^n0)
            twy = tw1y[j1+1] * ((-1)^m0)
            twz = tw1z[k1+1] * ((-1)^l0)
            push!(pruned_pts, PrunedPoint(
                Int32(i1+1), Int32(j1+1), Int32(k1+1),
                Int32(h[1]+1), Int32(h[2]+1), Int32(h[3]+1),
                twx, twy, twz))
        end
    end
    
    # Child needed set: {h mod M2 : h ∈ needed}
    child_needed_set = Set{NTuple{3,Int}}()
    for h in needed
        push!(child_needed_set, mod.(h, M2))
    end
    
    # S₃-aware child pruning: SP children produce S₃-symmetric output.
    # Only compute butterfly at S₃-canonical points; fill the rest via permutation.
    child_canonical_set = Set{NTuple{3,Int}}()
    s3_fill = S3FillPair[]
    for h in child_needed_set
        canon = s3_canonical(h[1], h[2], h[3], M2_1)
        push!(child_canonical_set, canon)
        if h != canon
            push!(s3_fill, S3FillPair(
                Int32(h[1]+1), Int32(h[2]+1), Int32(h[3]+1),
                Int32(canon[1]+1), Int32(canon[2]+1), Int32(canon[3]+1)))
        end
    end
    child_needed = collect(child_canonical_set)
    
    # Recurse on SP children (with S₃-reduced needed set)
    children_plans = Union{PrecompPlan, Nothing}[nothing, nothing]
    for (idx, child) in enumerate(node.children)
        if child !== nothing
            children_plans[idx] = build_precomp_plan(child, child_needed)
        end
    end
    
    return PrecompPlan(node, F100, F010, F101, F011, tw1x, tw1y, tw1z, 
                       use_full, pruned_pts, s3_fill, children_plans)
end

"""
    execute_precomp_recon!(result, pp)

Execute pruned P3c butterfly reconstruction using precomputed plan.
Only computes the output points specified during `build_precomp_plan`.
Assumes `execute_p3c_forward!` has been called on the P3c tree.
"""
function execute_precomp_recon!(result::Array{ComplexF64,3}, pp::PrecompPlan)
    node = pp.node_ref
    M = node.size; M2 = node.half
    M2_1 = M2[1]; M2_2 = M2[2]; M2_3 = M2[3]
    
    # Recurse on SP children (bottom-up)
    sp_freq = node.sp_bufs
    for (idx, cp) in enumerate(pp.children_plans)
        if cp !== nothing
            execute_precomp_recon!(sp_freq[idx], cp)
        end
    end
    
    # S₃ fill: populate non-canonical SP entries from canonical ones.
    # Both SP arrays (F000, F111) have S₃ symmetry.
    @inbounds for fp in pp.s3_fill
        sp_freq[1][fp.ti, fp.tj, fp.tk] = sp_freq[1][fp.si, fp.sj, fp.sk]
        sp_freq[2][fp.ti, fp.tj, fp.tk] = sp_freq[2][fp.si, fp.sj, fp.sk]
    end
    
    F000 = sp_freq[1]; F111 = sp_freq[2]
    F001 = node.gp_ffts[1]; F110 = node.gp_ffts[2]
    
    if pp.use_full
        # Full butterfly: build C₃-rotated copies for contiguous access
        F100 = pp.F100; F010 = pp.F010; F101 = pp.F101; F011 = pp.F011
        @inbounds for k in 1:M2_3, j in 1:M2_2, i in 1:M2_1
            F100[i,j,k] = F001[j,k,i]; F010[i,j,k] = F001[k,i,j]
            F101[i,j,k] = F110[k,i,j]; F011[i,j,k] = F110[j,k,i]
        end
        tw1x = pp.tw1x; tw1y = pp.tw1y; tw1z = pp.tw1z
        @inbounds for l0 in 0:1, m0 in 0:1, n0 in 0:1
            sx = (-1)^n0; sy = (-1)^m0; sz = (-1)^l0
            for k in 1:M2_3
                twz = tw1z[k] * sz
                for j in 1:M2_2
                    twy = tw1y[j] * sy; twyz = twy * twz
                    for i in 1:M2_1
                        twx = tw1x[i] * sx
                        val = F000[i,j,k] + twz*F001[i,j,k] + twy*F010[i,j,k] + twx*F100[i,j,k] +
                              twyz*F011[i,j,k] + twx*twz*F101[i,j,k] + twx*twy*F110[i,j,k] +
                              twx*twyz*F111[i,j,k]
                        result[i + n0*M2_1, j + m0*M2_2, k + l0*M2_3] = val
                    end
                end
            end
        end
    else
        # Pruned butterfly: inline C₃ rotation (no rotation copy needed)
        # F100[i,j,k] = F001[j,k,i]   F010[i,j,k] = F001[k,i,j]
        # F101[i,j,k] = F110[k,i,j]   F011[i,j,k] = F110[j,k,i]
        pts = pp.pruned_pts
        @inbounds for p in pts
            ii = p.i1p1; jj = p.j1p1; kk = p.k1p1
            twyz = p.twy * p.twz
            val = F000[ii,jj,kk] + 
                  p.twz * F001[ii,jj,kk] + 
                  p.twy * F001[kk,ii,jj] +          # F010
                  p.twx * F001[jj,kk,ii] +          # F100
                  twyz * F110[jj,kk,ii] +            # F011
                  p.twx*p.twz * F110[kk,ii,jj] +    # F101
                  p.twx*p.twy * F110[ii,jj,kk] +
                  p.twx*twyz * F111[ii,jj,kk]
            result[p.oi, p.oj, p.ok] = val
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────
# End-to-end execution
# ─────────────────────────────────────────────────────────────────────────

"""
    execute_staged!(plan, u) -> Array{ComplexF64, 3}

Execute the staged KRFFT: pack f₀₀₀ → P3c recursive decomposition → 
butterfly reconstruct F₀₀₀.

Returns F₀₀₀ (N³), from which the full (2N)³ spectrum is computable
via `lookup_full_spectrum`.
"""
function execute_staged!(plan::StagedPm3mPlan, u::AbstractArray{<:Real,3})
    pack_a8!(plan, u)
    execute_p3c_forward!(plan.p3c_root, plan.a8_buf)
    reconstruct_p3c!(plan.a8_fft, plan.p3c_root)
    return plan.a8_fft
end

"""
    lookup_full_spectrum(plan, F000, hx, hy, hz) -> ComplexF64

Compute F(h) for any h in the full (2N)³ grid from F₀₀₀,
using A8 symmetry: all 8 stride-2 subgrid FFTs derived from F₀₀₀.

For Pm-3m with b=1/2 shift, the 8 subgrid FFTs are:
  F_{000}(h₁) = F₀₀₀(i,j,k)
  F_{110}(h₁) = cispi(2(i+j)/N) · F₀₀₀(-i,-j,k)    [β]
  F_{101}(h₁) = cispi(2(i+k)/N) · F₀₀₀(-i,j,-k)    [γ]
  F_{111}(h₁) = cispi(2(i+j+k)/N) · F₀₀₀(-i,-j,-k)  [ν]
  F_{011}(h₁) = cispi(2(j+k)/N) · F₀₀₀(i,-j,-k)    [βγ]
  F_{001}(h₁) = cispi(2k/N) · F₀₀₀(i,j,-k)          [chain: β→ν]
  F_{100}(h₁) = cispi(2i/N) · F₀₀₀(-i,j,k)          [chain: βγ→ν]
  F_{010}(h₁) = cispi(2j/N) · F₀₀₀(i,-j,k)          [chain: γ→ν]
"""
function lookup_full_spectrum(plan::StagedPm3mPlan, F000::Array{ComplexF64,3}, 
                               hx::Int, hy::Int, hz::Int)
    N = plan.N_half; Nf = plan.N_full
    i1 = mod(hx, N[1]); j1 = mod(hy, N[2]); k1 = mod(hz, N[3])
    
    phi_x = cispi(-2 * hx / Nf[1])
    phi_y = cispi(-2 * hy / Nf[2])
    phi_z = cispi(-2 * hz / Nf[3])
    
    @inline f0(i,j,k) = F000[mod(i,N[1])+1, mod(j,N[2])+1, mod(k,N[3])+1]
    
    Ni = N[1]
    v000 = f0(i1, j1, k1)
    v110 = cispi(2(i1+j1)/Ni) * f0(-i1, -j1, k1)
    v101 = cispi(2(i1+k1)/Ni) * f0(-i1, j1, -k1)
    v111 = cispi(2(i1+j1+k1)/Ni) * f0(-i1, -j1, -k1)
    v011 = cispi(2(j1+k1)/Ni) * f0(i1, -j1, -k1)
    v001 = cispi(2k1/Ni) * f0(i1, j1, -k1)
    v100 = cispi(2i1/Ni) * f0(-i1, j1, k1)
    v010 = cispi(2j1/Ni) * f0(i1, -j1, k1)
    
    return v000 + 
        phi_z*v001 + phi_y*v010 + phi_x*v100 +
        phi_y*phi_z*v011 + phi_x*phi_z*v101 + phi_x*phi_y*v110 +
        phi_x*phi_y*phi_z*v111
end

"""
    s3_fill_output!(F000)

Fill the full (N/2)³ F₀₀₀ array using S₃ symmetry: F₀₀₀(perm(h)) = F₀₀₀(h).
After pruned butterfly computes S₃-canonical entries, this fills all S₃-related
entries so that A8 combination can read any index.
"""
function s3_fill_output!(F000::Array{ComplexF64,3})
    N = size(F000, 1)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        # S₃ canonical = sorted index (1-based)
        a, b, c = i, j, k
        if a > b; a, b = b, a; end
        if b > c; b, c = c, b; end
        if a > b; a, b = b, a; end
        if (a, b, c) != (i, j, k)
            F000[i, j, k] = F000[a, b, c]
        end
    end
end

"""Precomputed per-point data for A8 spectral combination.
Stores 8 F₀₀₀ linear indices and combined twiddle factors (A8 phase × full-grid phase)."""
struct A8Point
    idx000::Int32; idx110::Int32; idx101::Int32; idx111::Int32
    idx011::Int32; idx001::Int32; idx100::Int32; idx010::Int32
    tw000::ComplexF64; tw110::ComplexF64; tw101::ComplexF64; tw111::ComplexF64
    tw011::ComplexF64; tw001::ComplexF64; tw100::ComplexF64; tw010::ComplexF64
end

"""
    build_a8_points(plan, spec, F000_size) → Vector{A8Point}

Precompute A8 spectral combination data for all spectral ASU points.
Each `A8Point` stores linear indices into F₀₀₀ and combined twiddle factors,
so the inner loop is a simple 8-element multiply-accumulate with no transcendental calls.
"""
function build_a8_points(plan::StagedPm3mPlan, spec, F000_size::NTuple{3,Int})
    N = plan.N_half; Nf = plan.N_full
    Ni = N[1]; sz1 = F000_size[1]; sz2 = F000_size[2]
    n_spec = length(spec.points)
    
    tw_a8 = [cispi(2i / Ni) for i in 0:Ni-1]
    phi_x_tab = [cispi(-2i / Nf[1]) for i in 0:Nf[1]-1]
    phi_y_tab = [cispi(-2i / Nf[2]) for i in 0:Nf[2]-1]
    phi_z_tab = [cispi(-2i / Nf[3]) for i in 0:Nf[3]-1]
    
    points = Vector{A8Point}(undef, n_spec)
    
    for h_idx in 1:n_spec
        h = get_k_vector(spec, h_idx)
        hx, hy, hz = h
        i1 = mod(hx, Ni); j1 = mod(hy, Ni); k1 = mod(hz, Ni)
        
        @inline lidx(i, j, k) = mod(i, Ni) + 1 + sz1 * (mod(j, Ni) + sz2 * mod(k, Ni))
        
        idx000 = lidx(i1, j1, k1)
        idx110 = lidx(-i1, -j1, k1)
        idx101 = lidx(-i1, j1, -k1)
        idx111 = lidx(-i1, -j1, -k1)
        idx011 = lidx(i1, -j1, -k1)
        idx001 = lidx(i1, j1, -k1)
        idx100 = lidx(-i1, j1, k1)
        idx010 = lidx(i1, -j1, k1)
        
        ph_ij = tw_a8[mod(i1+j1, Ni)+1]
        ph_ik = tw_a8[mod(i1+k1, Ni)+1]
        ph_ijk = tw_a8[mod(i1+j1+k1, Ni)+1]
        ph_jk = tw_a8[mod(j1+k1, Ni)+1]
        ph_k = tw_a8[mod(k1, Ni)+1]
        ph_i = tw_a8[mod(i1, Ni)+1]
        ph_j = tw_a8[mod(j1, Ni)+1]
        
        phi_x = phi_x_tab[mod(hx, Nf[1])+1]
        phi_y = phi_y_tab[mod(hy, Nf[2])+1]
        phi_z = phi_z_tab[mod(hz, Nf[3])+1]
        
        xy = phi_x * phi_y; xz = phi_x * phi_z
        yz = phi_y * phi_z; xyz = xy * phi_z
        
        points[h_idx] = A8Point(
            Int32(idx000), Int32(idx110), Int32(idx101), Int32(idx111),
            Int32(idx011), Int32(idx001), Int32(idx100), Int32(idx010),
            one(ComplexF64), ph_ij*xy, ph_ik*xz, ph_ijk*xyz,
            ph_jk*yz, ph_k*phi_z, ph_i*phi_x, ph_j*phi_y)
    end
    return points
end

"""
    execute_a8_combination!(result, F000, a8_pts)

Execute precomputed A8 spectral combination: for each spectral ASU point,
compute F(h) = Σ tw × F₀₀₀[idx] using precomputed indices and twiddles.
"""
function execute_a8_combination!(result::Vector{ComplexF64}, F000::Array{ComplexF64,3}, 
                                  a8_pts::Vector{A8Point})
    @inbounds for i in eachindex(a8_pts)
        p = a8_pts[i]
        result[i] = p.tw000 * F000[p.idx000] + p.tw110 * F000[p.idx110] +
                     p.tw101 * F000[p.idx101] + p.tw111 * F000[p.idx111] +
                     p.tw011 * F000[p.idx011] + p.tw001 * F000[p.idx001] +
                     p.tw100 * F000[p.idx100] + p.tw010 * F000[p.idx010]
    end
    return result
end

"""
    compute_spectral_asu!(result, plan, F000, spec)

Compute F(h) at all spectral ASU points using `lookup_full_spectrum`.
This is the non-precomputed version (slower but useful for debugging).
For performance, use `build_a8_points` + `execute_a8_combination!` instead.
"""
function compute_spectral_asu!(result::Vector{ComplexF64}, plan::StagedPm3mPlan,
                                F000::Array{ComplexF64,3}, spec)
    n_spec = length(spec.points)
    @assert length(result) >= n_spec
    @inbounds for h_idx in 1:n_spec
        h = get_k_vector(spec, h_idx)
        result[h_idx] = lookup_full_spectrum(plan, F000, h...)
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────
# Targeted point query: bypass full reconstruction
# ─────────────────────────────────────────────────────────────────────────

"""
    query_p3c_point(node, ii, jj, kk) -> ComplexF64

Query a single F(ii,jj,kk) from the P3c tree without full reconstruction.
Bottom-up butterfly at this specific (ii,jj,kk) only.
Assumes execute_p3c_forward! has been called.
"""
function query_p3c_point(node::P3cNode, ii::Int, jj::Int, kk::Int)
    M = node.size; M2 = node.half
    i2 = mod(ii, M2[1]); j2 = mod(jj, M2[2]); k2 = mod(kk, M2[3])
    
    phi_x = cispi(-2 * ii / M[1])
    phi_y = cispi(-2 * jj / M[2])
    phi_z = cispi(-2 * kk / M[3])
    
    # SP sub-subgrids: recursively query or leaf lookup
    v000 = if node.children[1] !== nothing
        query_p3c_point(node.children[1], i2, j2, k2)
    else
        node.sp_bufs[1][i2+1, j2+1, k2+1]
    end
    
    v111 = if node.children[2] !== nothing
        query_p3c_point(node.children[2], i2, j2, k2)
    else
        node.sp_bufs[2][i2+1, j2+1, k2+1]
    end
    
    # GP sub-subgrids from stored FFTs + C₃ orbit
    F001 = node.gp_ffts[1]; F110 = node.gp_ffts[2]
    v001 = F001[i2+1, j2+1, k2+1]
    v100 = F001[j2+1, k2+1, i2+1]
    v010 = F001[k2+1, i2+1, j2+1]
    v110 = F110[i2+1, j2+1, k2+1]
    v101 = F110[k2+1, i2+1, j2+1]
    v011 = F110[j2+1, k2+1, i2+1]
    
    return v000 + 
        phi_z*v001 + phi_y*v010 + phi_x*v100 +
        phi_y*phi_z*v011 + phi_x*phi_z*v101 + phi_x*phi_y*v110 +
        phi_x*phi_y*phi_z*v111
end

"""
    lookup_spectrum_targeted(plan, hx, hy, hz) -> ComplexF64

Compute F(h) for any h in the full (2N)³ grid by:
1. Point-querying F₀₀₀(h₁) from the P3c tree (no full reconstruction)
2. A8 butterfly to combine all 8 subgrid contributions

Assumes execute_p3c_forward! has been called on plan.p3c_root.
"""
function lookup_spectrum_targeted(plan::StagedPm3mPlan, hx::Int, hy::Int, hz::Int)
    N = plan.N_half; Nf = plan.N_full; Ni = N[1]
    i1 = mod(hx, N[1]); j1 = mod(hy, N[2]); k1 = mod(hz, N[3])
    root = plan.p3c_root
    
    phi_x = cispi(-2 * hx / Nf[1])
    phi_y = cispi(-2 * hy / Nf[2])
    phi_z = cispi(-2 * hz / Nf[3])
    
    # Query F₀₀₀ at 8 different h₁ variants via A8 symmetry
    # Each query goes through the P3c tree (O(depth) per query)
    q(i,j,k) = query_p3c_point(root, mod(i,N[1]), mod(j,N[2]), mod(k,N[3]))
    
    v000 = q(i1, j1, k1)
    v110 = cispi(2(i1+j1)/Ni) * q(-i1, -j1, k1)
    v101 = cispi(2(i1+k1)/Ni) * q(-i1, j1, -k1)
    v111 = cispi(2(i1+j1+k1)/Ni) * q(-i1, -j1, -k1)
    v011 = cispi(2(j1+k1)/Ni) * q(i1, -j1, -k1)
    v001 = cispi(2k1/Ni) * q(i1, j1, -k1)
    v100 = cispi(2i1/Ni) * q(-i1, j1, k1)
    v010 = cispi(2j1/Ni) * q(i1, -j1, k1)
    
    return v000 + 
        phi_z*v001 + phi_y*v010 + phi_x*v100 +
        phi_y*phi_z*v011 + phi_x*phi_z*v101 + phi_x*phi_y*v110 +
        phi_x*phi_y*phi_z*v111
end

"""
    forward_and_query!(plan, u, queries) -> Vector{ComplexF64}

Combined pipeline: pack f₀₀₀ → P3c forward → per-point A8+P3c queries.
`queries` is a Vector of (hx,hy,hz) tuples for spectral ASU points.
No full-grid reconstruction is performed.
"""
function forward_and_query!(plan::StagedPm3mPlan, u::AbstractArray{<:Real,3}, 
                            queries::Vector{NTuple{3,Int}})
    pack_a8!(plan, u)
    execute_p3c_forward!(plan.p3c_root, plan.a8_buf)
    
    result = Vector{ComplexF64}(undef, length(queries))
    @inbounds for (idx, (hx,hy,hz)) in enumerate(queries)
        result[idx] = lookup_spectrum_targeted(plan, hx, hy, hz)
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────
# Convenience API (mirrors plan_krfft_g0asu / execute_g0asu_krfft!)
# ─────────────────────────────────────────────────────────────────────────

"""
    StagedPm3mFullPlan

Bundled plan for one-call execution of staged KRFFT on Pm-3m.
Created by `plan_staged_pm3m_full`, executed by `execute_staged_full!`.
"""
struct StagedPm3mFullPlan
    staged::StagedPm3mPlan
    precomp::PrecompPlan
    output::Array{ComplexF64,3}   # (N/2)³ preallocated output
end

"""
    plan_staged_pm3m_full(N_full; use_yx=false) → StagedPm3mFullPlan

Create a full staged KRFFT plan for Pm-3m.
`N_full` is the full grid size tuple `(N, N, N)`.
Precomputes the pruned butterfly for all `(N/2)³` spectral points.
"""
function plan_staged_pm3m_full(N_full::NTuple{3,Int}; use_yx::Bool=false)
    plan = plan_staged_pm3m(N_full; use_yx)
    N2 = N_full .÷ 2
    needed = NTuple{3,Int}[]
    for k in 0:N2[3]-1, j in 0:N2[2]-1, i in 0:N2[1]-1
        push!(needed, (i, j, k))
    end
    pp = build_precomp_plan(plan.p3c_root, needed)
    output = zeros(ComplexF64, N2...)
    return StagedPm3mFullPlan(plan, pp, output)
end

"""
    execute_staged_full!(fp, u) → Array{ComplexF64,3}

Execute staged KRFFT: pack f₀₀₀ → P3c forward → pruned butterfly reconstruction.
Returns F₀₀₀ as (N/2)³ array.

The full (2N)³ spectrum can be recovered via `lookup_full_spectrum`.
"""
function execute_staged_full!(fp::StagedPm3mFullPlan, u::AbstractArray{<:Real,3})
    pack_a8!(fp.staged, u)
    execute_p3c_forward!(fp.staged.p3c_root, fp.staged.a8_buf)
    execute_precomp_recon!(fp.output, fp.precomp)
    return fp.output
end

