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
    
    if M2[1] >= min_size && M2[1] % 2 == 0
        children = Union{YxNode,Nothing}[
            build_yx_tree(M2, level+1; min_size),
            build_yx_tree(M2, level+1; min_size)]
    else
        children = Union{YxNode,Nothing}[nothing, nothing]
    end
    
    return YxNode(level, M, M2, gp_buf, gp_fft, sp_bufs, fft_plan, children)
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
    
    return P3cNode(level, M, M2, gp_bufs, gp_ffts, gp_yx, sp_bufs, fft_plan, children)
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
            tmp = copy(node.sp_bufs[idx])
            mul!(node.sp_bufs[idx], node.fft_plan, tmp)
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
            tmp = copy(node.sp_bufs[idx])
            mul!(node.sp_bufs[idx], node.fft_plan, tmp)
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
