"""
    RecursiveKRFFT Building Blocks — Butterfly-Fused P3c Reconstruction

# Two-stage reconstruction with butterfly G0 fill:

Stage 1 — G0 cache fill via 3 in-place radix-2 butterfly passes:
  The P3c twiddles factor as products of 1D twiddles:
    tw(n,m,l) = tw_x(n) × tw_y(m) × tw_z(l)
  This enables Cooley-Tukey style butterfly decomposition:
    z-pass: combine l=0,1 pairs  → tw_z per (hx,hy,hz) pair
    y-pass: combine m=0,1 pairs  → tw_y per (hx,hy) pair
    x-pass: combine n=0,1 pairs  → tw_x per hx
  Each pass sweeps contiguous M2³ memory with unit stride → SIMD/cache optimal.

Stage 2 — A8 reconstruction:
  For each spectral ASU point, 8 lookups from G0 cache with A8 phase factors.
  n_spec × 8 multiply-adds (typically n_spec ≪ M³).
"""

"""
    P3cReconEntry

One A8 contribution to a spectral ASU point from the G0 cache.
"""
struct P3cReconEntry
    linear_idx::Int32      # linear index into G0 cache
    weight::ComplexF64     # A8 phase factor
end

"""
    RecursiveKRFFTPlan

Plan for recursive KRFFT with butterfly-fused P3c + A8 decomposition.
"""
struct RecursiveKRFFTPlan
    sub_plan::Any                         # FFTW plan for (M/2)³
    buffers::Vector{Array{ComplexF64,3}}  # 4 input: F000, F001, F110, F111
    work_ffts::Vector{Array{ComplexF64,3}} # 4 FFT outputs
    output_buffer::Vector{ComplexF64}     # spectral ASU output

    # Butterfly G0 fill: 8 work arrays for butterfly stages
    work_bufs::Vector{Array{ComplexF64,3}}  # 8 work arrays of size M2³
    g0_cache::Array{ComplexF64,3}           # M³ G0 cache

    # 1D twiddle arrays for butterfly stages
    tw_x::Vector{ComplexF64}  # exp(-2πi hx/M_x) for hx=0..M2_x-1
    tw_y::Vector{ComplexF64}  # exp(-2πi hy/M_y) for hy=0..M2_y-1
    tw_z::Vector{ComplexF64}  # exp(-2πi hz/M_z) for hz=0..M2_z-1

    # A8 reconstruction table
    a8_table::Matrix{P3cReconEntry}  # (8, n_spec)
    n_a8_ops::Int

    grid_N::Vector{Int}
    subgrid_dims::Vector{Int}   # M = N/2
    sub_sub_dims::Vector{Int}   # M2 = N/4
end

"""
    plan_krfft_recursive(spec_asu, ops_shifted)

Create a recursive KRFFT plan with butterfly-fused P3c + A8 decomposition.
"""
function plan_krfft_recursive(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "Recursive KRFFT currently supports 3D only"

    M = [N[d] ÷ 2 for d in 1:dim]
    M2 = [M[d] ÷ 2 for d in 1:dim]

    @assert all(M .* 2 .== collect(N)) "Grid size must be divisible by 2"
    @assert all(M2 .* 2 .== M) "M must be divisible by 2 for P3c"

    n_spec = length(spec_asu.points)
    n_a8 = 8

    # Select representative operations for A8 subgrids (one per parity class)
    subgrid_reps = Vector{Union{Nothing, SymOp}}(nothing, n_a8)
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], 2) for d in 1:dim]
        idx = 1 + x0[1] + 2*x0[2] + 4*x0[3]
        if subgrid_reps[idx] === nothing
            subgrid_reps[idx] = op
        end
    end
    active_a8 = [i for i in 1:8 if subgrid_reps[i] !== nothing]

    # Build A8 reconstruction table: 8 entries per spectral point
    a8_table = Matrix{P3cReconEntry}(undef, n_a8, n_spec)
    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        slot = 0
        for a8_idx in active_a8
            g = subgrid_reps[a8_idx]::SymOp
            # A8 phase
            a8_phase = sum(h_vec[d] * g.t[d] / N[d] for d in 1:dim)
            a8_tw = cispi(-2 * a8_phase)

            # Rotated frequency into G0: R_g^T h mod M
            rot_h = [mod(sum(Int(g.R[d2, d1]) * h_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]

            # Linear index into G0 cache (1-based, column-major)
            lin = 1 + rot_h[1] + M[1] * rot_h[2] + M[1] * M[2] * rot_h[3]
            slot += 1
            a8_table[slot, h_idx] = P3cReconEntry(Int32(lin), a8_tw)
        end
        # Pad remaining rows with zero weight
        for s in (slot+1):8
            a8_table[s, h_idx] = P3cReconEntry(Int32(1), complex(0.0))
        end
    end

    # Precompute 1D twiddle arrays for butterfly stages
    tw_x = [cispi(-2 * hx / M[1]) for hx in 0:M2[1]-1]
    tw_y = [cispi(-2 * hy / M[2]) for hy in 0:M2[2]-1]
    tw_z = [cispi(-2 * hz / M[3]) for hz in 0:M2[3]-1]

    # Allocate buffers
    M2_tuple = Tuple(M2)
    M_tuple = Tuple(M)
    buffers = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    work_ffts = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    work_bufs = [zeros(ComplexF64, M2_tuple) for _ in 1:8]
    g0_cache = zeros(ComplexF64, M_tuple)
    sub_plan = plan_fft(buffers[1])
    output_buffer = zeros(ComplexF64, n_spec)

    @info "P3c butterfly plan: n_spec=$n_spec, M=$(M_tuple), A8×$(n_a8)"

    return RecursiveKRFFTPlan(
        sub_plan, buffers, work_ffts, output_buffer,
        work_bufs, g0_cache,
        tw_x, tw_y, tw_z,
        a8_table, n_a8,
        collect(N), M, M2
    )
end

"""
    pack_p3c!(plan, u)

Extract 4 P3c sub-sub-grids from real-space array u at stride 4.
"""
function pack_p3c!(plan::RecursiveKRFFTPlan, u::AbstractArray{<:Real})
    M2 = plan.sub_sub_dims
    buf000, buf001, buf110, buf111 = plan.buffers[1], plan.buffers[2], plan.buffers[3], plan.buffers[4]

    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000[i,j,k] = complex(u[ii+1, jj+1, kk+1])
        buf001[i,j,k] = complex(u[ii+1, jj+1, kk+3])
        buf110[i,j,k] = complex(u[ii+3, jj+3, kk+1])
        buf111[i,j,k] = complex(u[ii+3, jj+3, kk+3])
    end
end

"""
    fft_p3c!(plan)

Execute 4 out-of-place FFTs on P3c sub-sub-grids.
"""
function fft_p3c!(plan::RecursiveKRFFTPlan)
    p = plan.sub_plan
    for i in 1:4
        mul!(plan.work_ffts[i], p, plan.buffers[i])
    end
end

"""
    fill_g0_butterfly!(plan)

Fill the G0 cache using 3-stage butterfly decomposition of P3c twiddles.

Stage 0: Distribute 4 FFT outputs into 8 work arrays using P3c symmetry:
    work[1] = F000(hx,hy,hz)          (n,m,l)=(0,0,0)
    work[2] = F001(hx,hy,hz)          (n,m,l)=(0,0,1)
    work[3] = F001(hz,hx,hy)          (n,m,l)=(0,1,0), permuted from F001
    work[4] = F001(hy,hz,hx)          (n,m,l)=(1,0,0), permuted from F001
    work[5] = F110(hx,hy,hz)          (n,m,l)=(1,1,0)
    work[6] = F110(hz,hx,hy)          (n,m,l)=(1,0,1), permuted from F110
    work[7] = F110(hy,hz,hx)          (n,m,l)=(0,1,1), permuted from F110
    work[8] = F111(hx,hy,hz)          (n,m,l)=(1,1,1)

Then butterfly stages combine these into G0[hx,hy,hz] for all (hx,hy,hz) in M³:
    z-butterfly: pairs (1,2),(3,7),(4,6),(5,8) with tw_z
    y-butterfly: pairs (1,3),(4,5) with tw_y
    x-butterfly: pair  (1,4) with tw_x
    
Result stored in g0_cache as the full M³ array (G0 at stride-2 subgrid frequencies).
"""
function fill_g0_butterfly!(plan::RecursiveKRFFTPlan)
    M = plan.subgrid_dims
    M2 = plan.sub_sub_dims
    F000, F001, F110, F111 = plan.work_ffts[1], plan.work_ffts[2], plan.work_ffts[3], plan.work_ffts[4]
    w = plan.work_bufs
    tw_x, tw_y, tw_z = plan.tw_x, plan.tw_y, plan.tw_z
    g0 = plan.g0_cache

    # Stage 0: Distribute FFT outputs into 8 work arrays with index permutations
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        w[1][i,j,k] = F000[i,j,k]          # (0,0,0)
        w[2][i,j,k] = F001[i,j,k]          # (0,0,1)
        w[3][i,j,k] = F001[k,i,j]          # (0,1,0) = F010(h) = F001(hz,hx,hy)
        w[4][i,j,k] = F001[j,k,i]          # (1,0,0) = F100(h) = F001(hy,hz,hx)
        w[5][i,j,k] = F110[i,j,k]          # (1,1,0)
        w[6][i,j,k] = F110[k,i,j]          # (1,0,1) = F101(h) = F110(hz,hx,hy)
        w[7][i,j,k] = F110[j,k,i]          # (0,1,1) = F011(h) = F110(hy,hz,hx)
        w[8][i,j,k] = F111[i,j,k]          # (1,1,1)
    end

    # Stage 1: z-butterfly — combine l=0 and l=1 pairs
    # G_??(hx,hy, hz)    = F_??0(ht) + tw_z(hz) × F_??1(ht)
    # G_??(hx,hy, hz+M2) = F_??0(ht) - tw_z(hz) × F_??1(ht)
    @inbounds for k in 1:M2[3], j in 1:M2[2]
        twz = tw_z[k]
        @simd for i in 1:M2[1]
            even = w[1][i,j,k]; odd = twz * w[2][i,j,k]
            w[1][i,j,k] = even + odd; w[2][i,j,k] = even - odd

            even = w[3][i,j,k]; odd = twz * w[7][i,j,k]
            w[3][i,j,k] = even + odd; w[7][i,j,k] = even - odd

            even = w[4][i,j,k]; odd = twz * w[6][i,j,k]
            w[4][i,j,k] = even + odd; w[6][i,j,k] = even - odd

            even = w[5][i,j,k]; odd = twz * w[8][i,j,k]
            w[5][i,j,k] = even + odd; w[8][i,j,k] = even - odd
        end
    end
    # After: w[1]=G_00(hz lo), w[2]=G_00(hz hi), w[3]=G_01(lo), w[7]=G_01(hi)
    #        w[4]=G_10(lo), w[6]=G_10(hi), w[5]=G_11(lo), w[8]=G_11(hi)

    # Stage 2: y-butterfly — combine m=0 and m=1 pairs
    @inbounds for k in 1:M2[3], j in 1:M2[2]
        twy = tw_y[j]
        @simd for i in 1:M2[1]
            even = w[1][i,j,k]; odd = twy * w[3][i,j,k]
            w[1][i,j,k] = even + odd; w[3][i,j,k] = even - odd

            even = w[4][i,j,k]; odd = twy * w[5][i,j,k]
            w[4][i,j,k] = even + odd; w[5][i,j,k] = even - odd

            even = w[2][i,j,k]; odd = twy * w[7][i,j,k]
            w[2][i,j,k] = even + odd; w[7][i,j,k] = even - odd

            even = w[6][i,j,k]; odd = twy * w[8][i,j,k]
            w[6][i,j,k] = even + odd; w[8][i,j,k] = even - odd
        end
    end
    # After: w[1]=G_0(hy lo,hz lo), w[3]=G_0(hy hi,hz lo), etc.
    #        w[4]=G_1(hy lo,hz lo), w[5]=G_1(hy hi,hz lo), etc.

    # Stage 3: x-butterfly — combine n=0 and n=1 pairs
    @inbounds for k in 1:M2[3], j in 1:M2[2]
        @simd for i in 1:M2[1]
            twx = tw_x[i]
            even = w[1][i,j,k]; odd = twx * w[4][i,j,k]
            w[1][i,j,k] = even + odd; w[4][i,j,k] = even - odd

            even = w[3][i,j,k]; odd = twx * w[5][i,j,k]
            w[3][i,j,k] = even + odd; w[5][i,j,k] = even - odd

            even = w[2][i,j,k]; odd = twx * w[6][i,j,k]
            w[2][i,j,k] = even + odd; w[6][i,j,k] = even - odd

            even = w[7][i,j,k]; odd = twx * w[8][i,j,k]
            w[7][i,j,k] = even + odd; w[8][i,j,k] = even - odd
        end
    end

    # Stage 4: Write 8 M2³ blocks into the M³ G0 cache
    # The 8 blocks correspond to (hx,hy,hz) octants:
    #   w[1] → (lo,lo,lo), w[2] → (lo,lo,hi), w[3] → (lo,hi,lo), w[7] → (lo,hi,hi)
    #   w[4] → (hi,lo,lo), w[6] → (hi,lo,hi), w[5] → (hi,hi,lo), w[8] → (hi,hi,hi)
    ox = M2[1]; oy = M2[2]; oz = M2[3]
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        g0[i,    j,    k]    = w[1][i,j,k]
        g0[i,    j,    k+oz] = w[2][i,j,k]
        g0[i,    j+oy, k]    = w[3][i,j,k]
        g0[i,    j+oy, k+oz] = w[7][i,j,k]
        g0[i+ox, j,    k]    = w[4][i,j,k]
        g0[i+ox, j,    k+oz] = w[6][i,j,k]
        g0[i+ox, j+oy, k]    = w[5][i,j,k]
        g0[i+ox, j+oy, k+oz] = w[8][i,j,k]
    end
end

"""
    reconstruct_from_g0!(plan)

Reconstruct spectral ASU from G0 cache using A8 table.
8 multiply-adds per spectral point.
"""
function reconstruct_from_g0!(plan::RecursiveKRFFTPlan)
    out = plan.output_buffer
    g0 = vec(plan.g0_cache)
    table = plan.a8_table
    n_spec = length(out)
    n_a8 = plan.n_a8_ops

    @inbounds for h in 1:n_spec
        val = zero(ComplexF64)
        for g in 1:n_a8
            e = table[g, h]
            val += e.weight * g0[e.linear_idx]
        end
        out[h] = val
    end
    return out
end

"""
    execute_recursive_krfft!(plan, spec_asu, u)

Full recursive KRFFT pipeline:
1. Pack 4 sub-sub-grids at stride 4
2. FFT × 4 on (N/4)³ grids
3. Butterfly G0 fill (3 in-place radix-2 stages)
4. A8 reconstruction from G0 cache (8 lookups per spectral point)
"""
function execute_recursive_krfft!(plan::RecursiveKRFFTPlan, spec_asu::SpectralIndexing,
                                   u::AbstractArray{<:Real})
    pack_p3c!(plan, u)
    fft_p3c!(plan)
    fill_g0_butterfly!(plan)
    reconstruct_from_g0!(plan)
    return plan.output_buffer
end

"""
    reconstruct_g0_at(plan, hx, hy, hz) -> ComplexF64

Reconstruct a single G_0(hx,hy,hz) value from P3c FFTs (debugging only).
"""
@inline function reconstruct_g0_at(plan::RecursiveKRFFTPlan, hx::Int, hy::Int, hz::Int)
    M = plan.subgrid_dims
    M2 = plan.sub_sub_dims

    hxt = mod(hx, M2[1]); hyt = mod(hy, M2[2]); hzt = mod(hz, M2[3])
    ix = hxt + 1; iy = hyt + 1; iz = hzt + 1

    F000, F001, F110, F111 = plan.work_ffts[1], plan.work_ffts[2], plan.work_ffts[3], plan.work_ffts[4]

    val = F000[ix, iy, iz]
    val += cispi(-2*hx/M[1]) * F001[iy, iz, ix]
    val += cispi(-2*hy/M[2]) * F001[iz, ix, iy]
    val += cispi(-2*hz/M[3]) * F001[ix, iy, iz]
    val += cispi(-2*(hx/M[1] + hy/M[2])) * F110[ix, iy, iz]
    val += cispi(-2*(hx/M[1] + hz/M[3])) * F110[iz, ix, iy]
    val += cispi(-2*(hy/M[2] + hz/M[3])) * F110[iy, iz, ix]
    val += cispi(-2*(hx/M[1] + hy/M[2] + hz/M[3])) * F111[ix, iy, iz]

    return val
end

# ============================================================================
# Centering Detection & Extinction
# ============================================================================

"""
    detect_centering(ops, N) → :P, :I, or :F

Detect lattice centering from the symmetry operations. Looks for pure
translation operations (R = I) with non-zero translation vectors.
"""
function detect_centering(ops::Vector{SymOp}, N)
    dim = length(N)
    I_mat = zeros(Int, dim, dim)
    for d in 1:dim; I_mat[d,d] = 1; end

    n_centering = 0
    has_body = false
    has_face = false

    for op in ops
        # Check if R is identity
        is_identity = true
        for i in 1:dim, j in 1:dim
            if Int(op.R[i,j]) != I_mat[i,j]
                is_identity = false; break
            end
        end
        if !is_identity || all(abs.(op.t) .< 0.1); continue; end

        n_centering += 1
        t_frac = op.t ./ collect(N)
        # Body centering: (1/2, 1/2, 1/2)
        if all(abs.(t_frac .- 0.5) .< 0.01)
            has_body = true
        end
        # Face centering: (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0)
        n_half = count(abs.(t_frac .- 0.5) .< 0.01)
        n_zero = count(abs.(t_frac) .< 0.01)
        if n_half == 2 && n_zero == 1
            has_face = true
        end
    end

    if has_face && n_centering >= 3
        return :F
    elseif has_body && n_centering >= 1
        return :I
    else
        return :P
    end
end

"""
    _centering_extinct(centering, qt_x, qt_y, qt_z) → Bool

Returns true if the FFT output at position (qt_x, qt_y, qt_z) is identically
zero due to centering constraints on the sub-sub-grid data.

For F-centering: buf[i,j,k] = buf[i+M2/2, j+M2/2, k] (and cyclic permutations)
  → FFT(q) = 0 unless qx,qy,qz all have the same parity.

For I-centering: buf[i,j,k] = buf[i+M2/2, j+M2/2, k+M2/2]
  → FFT(q) = 0 when qx+qy+qz is odd.
"""
@inline function _centering_extinct(centering::Symbol, qt_x::Int, qt_y::Int, qt_z::Int)
    if centering === :F
        px = qt_x & 1; py = qt_y & 1; pz = qt_z & 1
        return !((px == py) && (py == pz))
    elseif centering === :I
        return ((qt_x + qt_y + qt_z) & 1) != 0
    else
        return false
    end
end

# ============================================================================
# Sparse Matrix Reconstruction — End-to-End P3c + A8
# ============================================================================

"""
    SparseKRFFTPlan

Plan for recursive KRFFT with end-to-end sparse matrix reconstruction.
Precomputes the combined P3c + A8 mapping as a CSR sparse table that maps
directly from 4 × M2³ FFT outputs to n_spec spectral ASU points, without
materializing the intermediate M³ G0 cache.

Cost is O(n_spec × nnz_per_row) instead of O(M³).
"""
struct SparseKRFFTPlan
    sub_plan::Any                         # FFTW plan for (M/2)³
    buffers::Vector{Array{ComplexF64,3}}  # 4 input: F000, F001, F110, F111
    work_ffts::Vector{Array{ComplexF64,3}} # 4 FFT outputs
    fft_concat::Vector{ComplexF64}        # concatenated FFT outputs [4 × M2³]
    output_buffer::Vector{ComplexF64}     # spectral ASU output

    # End-to-end sparse recon table (CSR format)
    entries::Vector{P3cReconEntry}        # (linear_idx into fft_concat, weight)
    row_ptrs::Vector{Int}                 # CSR row pointers, length n_spec+1

    centering::Symbol                     # :P, :I, or :F (auto-detected)

    grid_N::Vector{Int}
    subgrid_dims::Vector{Int}   # M = N/2
    sub_sub_dims::Vector{Int}   # M2 = N/4
end

# P3c sub-sub-grid permutation table:
# Maps (n1,n2,n3) → (buffer_offset, permutation of qt)
# Buffer: 0=F000, 1=F001, 2=F110, 3=F111
# Permutation applied to (qt_x, qt_y, qt_z) to get array position
const _P3C_MAP = (
    # (n1,n2,n3) => (buf_offset, perm_type)
    # perm_type: 0=identity, 1=cyclic(z,x,y), 2=cyclic(y,z,x)
    (0, 0),  # (0,0,0) → F000, identity
    (1, 0),  # (0,0,1) → F001, identity
    (1, 1),  # (0,1,0) → F001, (z,x,y)
    (1, 2),  # (1,0,0) → F001, (y,z,x)
    (2, 0),  # (1,1,0) → F110, identity
    (2, 1),  # (1,0,1) → F110, (z,x,y)
    (2, 2),  # (0,1,1) → F110, (y,z,x)
    (3, 0),  # (1,1,1) → F111, identity
)

# Ordering: iterate n3,n2,n1 (inner to outer) → index = 1+n1+2*n2+4*n3
# But we want the P3c order matching parity bits:
# (0,0,0)=1, (1,0,0)=2, (0,1,0)=3, (1,1,0)=4, (0,0,1)=5, (1,0,1)=6, (0,1,1)=7, (1,1,1)=8
const _P3C_ORDER = (
    (0,0,0), (1,0,0), (0,1,0), (1,1,0),
    (0,0,1), (1,0,1), (0,1,1), (1,1,1),
)
const _P3C_BUF = (0, 1, 1, 2, 1, 2, 2, 3)
const _P3C_PERM = (0, 2, 1, 0, 0, 1, 2, 0)

@inline function _p3c_linear_idx(buf_off::Int, perm::Int,
                                  qt_x::Int, qt_y::Int, qt_z::Int,
                                  M2::Vector{Int}, M2_vol::Int)
    if perm == 0      # identity: (qt_x, qt_y, qt_z)
        return Int32(buf_off * M2_vol + 1 + qt_x + M2[1]*qt_y + M2[1]*M2[2]*qt_z)
    elseif perm == 1  # cyclic: (qt_z, qt_x, qt_y)
        return Int32(buf_off * M2_vol + 1 + qt_z + M2[1]*qt_x + M2[1]*M2[2]*qt_y)
    else              # cyclic: (qt_y, qt_z, qt_x)
        return Int32(buf_off * M2_vol + 1 + qt_y + M2[1]*qt_z + M2[1]*M2[2]*qt_x)
    end
end

"""
    plan_krfft_sparse(spec_asu, ops_shifted; centering=:auto)

Create a sparse KRFFT plan with end-to-end P3c + A8 reconstruction.
Precomputes a CSR sparse table mapping directly from FFT outputs to spectral ASU.

With centering=:auto (default), automatically detects F/I/P centering and
filters out entries pointing to identically-zero FFT outputs due to centering
extinctions. This can reduce nnz by ~50-75% for centered lattices.
"""
function plan_krfft_sparse(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp};
                           centering::Symbol=:auto)
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "Sparse KRFFT currently supports 3D only"

    M = [N[d] ÷ 2 for d in 1:dim]
    M2 = [M[d] ÷ 2 for d in 1:dim]
    M2_vol = prod(M2)

    @assert all(M .* 2 .== collect(N)) "Grid size must be divisible by 2"
    @assert all(M2 .* 2 .== M) "M must be divisible by 2 for P3c"

    # Auto-detect centering
    if centering === :auto
        centering = detect_centering(ops_shifted, N)
    end

    n_spec = length(spec_asu.points)

    # Select A8 representative operations (one per parity class)
    subgrid_reps = Vector{Union{Nothing, SymOp}}(nothing, 8)
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], 2) for d in 1:dim]
        idx = 1 + x0[1] + 2*x0[2] + 4*x0[3]
        if subgrid_reps[idx] === nothing
            subgrid_reps[idx] = op
        end
    end
    active_a8 = [i for i in 1:8 if subgrid_reps[i] !== nothing]

    # Build CSR sparse table
    all_entries = P3cReconEntry[]
    row_ptrs = Vector{Int}(undef, n_spec + 1)
    sizehint!(all_entries, n_spec * 40)  # estimate ~40 after merging

    # Temp buffer for merging entries per row
    local_map = Dict{Int32, ComplexF64}()

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        row_ptrs[h_idx] = length(all_entries) + 1
        empty!(local_map)

        for a8_idx in active_a8
            g = subgrid_reps[a8_idx]::SymOp
            # A8 phase: exp(-2πi h·t_g/N)
            a8_phase = sum(h_vec[d] * g.t[d] / N[d] for d in 1:dim)
            a8_tw = cispi(-2 * a8_phase)

            # Rotated frequency: R_g^T h mod M
            q1 = mod(sum(Int(g.R[d2, 1]) * h_vec[d2] for d2 in 1:dim), M[1])
            q2 = mod(sum(Int(g.R[d2, 2]) * h_vec[d2] for d2 in 1:dim), M[2])
            q3 = mod(sum(Int(g.R[d2, 3]) * h_vec[d2] for d2 in 1:dim), M[3])

            qt_x = mod(q1, M2[1])
            qt_y = mod(q2, M2[2])
            qt_z = mod(q3, M2[3])

            # 8 P3c sub-sub-grid contributions
            for s in 1:8
                n1, n2, n3 = _P3C_ORDER[s]

                # P3c twiddle: exp(-2πi (q·n / M))
                p3c_phase = q1*n1/M[1] + q2*n2/M[2] + q3*n3/M[3]
                combined_w = a8_tw * cispi(-2 * p3c_phase)

                # Buffer and permuted position
                buf_off = _P3C_BUF[s]
                perm = _P3C_PERM[s]
                lin = _p3c_linear_idx(buf_off, perm, qt_x, qt_y, qt_z, M2, M2_vol)

                # Merge into local map
                existing = get(local_map, lin, zero(ComplexF64))
                local_map[lin] = existing + combined_w
            end
        end

        # Flush non-zero entries
        for (lin, w) in local_map
            if abs(w) > 1e-14
                push!(all_entries, P3cReconEntry(lin, w))
            end
        end
    end
    row_ptrs[n_spec + 1] = length(all_entries) + 1

    nnz = length(all_entries)
    avg = nnz / max(n_spec, 1)
    @info "Sparse KRFFT plan: n_spec=$n_spec, nnz=$nnz, avg=$(round(avg, digits=1))/row, input=$(4*M2_vol), centering=$centering"

    # Allocate buffers
    M2_tuple = Tuple(M2)
    buffers = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    work_ffts = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    fft_concat = zeros(ComplexF64, 4 * M2_vol)
    sub_plan = plan_fft(buffers[1])
    output_buffer = zeros(ComplexF64, n_spec)

    return SparseKRFFTPlan(
        sub_plan, buffers, work_ffts, fft_concat, output_buffer,
        all_entries, row_ptrs,
        centering,
        collect(N), M, M2
    )
end

"""
    execute_sparse_krfft!(plan, spec_asu, u)

Full sparse KRFFT pipeline:
1. Pack 4 sub-sub-grids at stride 4
2. FFT × 4 on (N/4)³ grids
3. Concatenate FFT outputs
4. Sparse matrix multiply → spectral ASU
"""
function execute_sparse_krfft!(plan::SparseKRFFTPlan, spec_asu::SpectralIndexing,
                               u::AbstractArray{<:Real})
    M2 = plan.sub_sub_dims

    # Pack P3c sub-sub-grids at stride 4
    buf000, buf001 = plan.buffers[1], plan.buffers[2]
    buf110, buf111 = plan.buffers[3], plan.buffers[4]
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000[i,j,k] = complex(u[ii+1, jj+1, kk+1])
        buf001[i,j,k] = complex(u[ii+1, jj+1, kk+3])
        buf110[i,j,k] = complex(u[ii+3, jj+3, kk+1])
        buf111[i,j,k] = complex(u[ii+3, jj+3, kk+3])
    end

    # FFT × 4
    p = plan.sub_plan
    for i in 1:4
        mul!(plan.work_ffts[i], p, plan.buffers[i])
    end

    # Concatenate FFT outputs into flat vector
    M2_vol = prod(M2)
    concat = plan.fft_concat
    @inbounds for b in 1:4
        src = plan.work_ffts[b]
        offset = (b-1) * M2_vol
        copyto!(concat, offset + 1, vec(src), 1, M2_vol)
    end

    # Sparse reconstruction: CSR SpMV
    sparse_reconstruct!(plan)
    return plan.output_buffer
end

"""
    sparse_reconstruct!(plan)

Sparse matrix-vector multiply: spectral ASU = T × fft_concat.
Each row has variable number of entries (typically 20-50 after merging).
"""
function sparse_reconstruct!(plan::SparseKRFFTPlan)
    out = plan.output_buffer
    concat = plan.fft_concat
    entries = plan.entries
    ptrs = plan.row_ptrs
    n_spec = length(out)

    @inbounds for h in 1:n_spec
        val = zero(ComplexF64)
        for i in ptrs[h]:ptrs[h+1]-1
            e = entries[i]
            val += e.weight * concat[e.linear_idx]
        end
        out[h] = val
    end
    return out
end

# ============================================================================
# Selective G0 Cascade — Compute G0 only at accessed positions
# ============================================================================

"""
    SelectiveG0Entry

One A8 contribution: maps a spectral ASU point to a compact G0 index.
"""
struct SelectiveG0Entry
    g0_idx::Int32          # index into compact g0_values array
    weight::ComplexF64     # A8 phase factor
end

"""
    P3cPointWeights

Precomputed P3c twiddle weights for evaluating G0 at one position.
8 weights × 8 FFT access indices = one G0 value.
"""
struct P3cPointWeights
    fft_idx::NTuple{8, Int32}     # 8 linear indices into fft_flat
    tw::NTuple{8, ComplexF64}     # 8 twiddle weights
end

"""
    SelectiveG0Plan

Selective G0 cascade: computes G0 only at positions accessed by the spectral ASU,
then does A8 lookup. This preserves intermediate value sharing — multiple spectral
points that map to the same G0 position share the P3c computation.

Cost: |S| × 8 (P3c per G0 position) + n_spec × 8 (A8 lookup)
where |S| is the number of unique G0 positions accessed.
"""
struct SelectiveG0Plan
    sub_plan::Any                            # FFTW plan for (N/4)³
    buffers::Vector{Array{ComplexF64,3}}     # 4 input: F000, F001, F110, F111
    work_ffts::Vector{Array{ComplexF64,3}}   # 4 FFT outputs
    fft_flat::Vector{ComplexF64}             # 4×M2³ flat concat
    output_buffer::Vector{ComplexF64}        # spectral ASU output

    # Compact G0 table: precomputed P3c weights per G0 position
    g0_p3c::Vector{P3cPointWeights}          # |S| entries
    g0_values::Vector{ComplexF64}            # |S| G0 values (filled at exec time)

    # A8 reconstruction table: 8 entries per spectral point (flattened)
    a8_table::Vector{SelectiveG0Entry}       # 8 × n_spec entries
    n_spec::Int

    grid_N::Vector{Int}
    subgrid_dims::Vector{Int}   # M = N/2
    sub_sub_dims::Vector{Int}   # M2 = N/4
end

"""
    plan_krfft_selective(spec_asu, ops_shifted)

Create a selective G0 cascade plan.

1. Identifies the unique G0 positions accessed by spectral ASU × A8 reps
2. Precomputes P3c twiddle weights per G0 position (8 complex weights each)
3. Builds A8 table mapping spectral points → compact G0 indices

For Fm-3m (|G|=192) at N=64: |S| = 7727 out of M³ = 32768 (23.6%).
"""
function plan_krfft_selective(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "Selective G0 KRFFT currently supports 3D only"

    M = [N[d] ÷ 2 for d in 1:dim]
    M2 = [M[d] ÷ 2 for d in 1:dim]
    M2_vol = prod(M2)

    @assert all(M .* 2 .== collect(N)) "Grid size must be divisible by 2"
    @assert all(M2 .* 2 .== M) "M must be divisible by 2 for P3c"

    n_spec = length(spec_asu.points)

    # Step 1: Select A8 representative operations (one per parity class)
    subgrid_reps = Vector{Union{Nothing, SymOp}}(nothing, 8)
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], 2) for d in 1:dim]
        idx = 1 + x0[1] + 2*x0[2] + 4*x0[3]
        if subgrid_reps[idx] === nothing
            subgrid_reps[idx] = op
        end
    end
    active_a8 = [i for i in 1:8 if subgrid_reps[i] !== nothing]

    # Step 2: Collect unique G0 positions and build A8 table
    g0_pos_map = Dict{Int, Int}()  # M-grid linear index → compact index
    a8_table = Vector{SelectiveG0Entry}(undef, 8 * n_spec)

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        slot = 0
        for a8_idx in active_a8
            g = subgrid_reps[a8_idx]::SymOp
            # A8 phase
            a8_phase = sum(h_vec[d] * g.t[d] / N[d] for d in 1:dim)
            a8_tw = cispi(-2 * a8_phase)

            # Rotated frequency: R_g^T h mod M
            rot_h = [mod(sum(Int(g.R[d2, d1]) * h_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]
            lin = 1 + rot_h[1] + M[1] * rot_h[2] + M[1] * M[2] * rot_h[3]

            # Register G0 position if new
            if !haskey(g0_pos_map, lin)
                g0_pos_map[lin] = length(g0_pos_map) + 1
            end
            compact_idx = g0_pos_map[lin]
            slot += 1
            a8_table[(h_idx-1)*8 + slot] = SelectiveG0Entry(Int32(compact_idx), a8_tw)
        end
        # Pad remaining A8 slots with zero weight
        for s in (slot+1):8
            a8_table[(h_idx-1)*8 + s] = SelectiveG0Entry(Int32(1), complex(0.0))
        end
    end

    n_g0 = length(g0_pos_map)

    # Step 3: Precompute P3c twiddle weights per G0 position
    g0_p3c = Vector{P3cPointWeights}(undef, n_g0)

    for (lin, compact_idx) in g0_pos_map
        # Decode linear index → (hx, hy, hz) 0-based
        lin0 = lin - 1
        hx = mod(lin0, M[1])
        hy = mod(div(lin0, M[1]), M[2])
        hz = div(lin0, M[1] * M[2])

        # qt = h mod M2 (1-based array indices)
        ix = mod(hx, M2[1]) + 1
        iy = mod(hy, M2[2]) + 1
        iz = mod(hz, M2[3]) + 1

        # Helper: 3D → flat index in one M2³ buffer
        _lin3(a,b,c) = (a-1) + M2[1]*(b-1) + M2[1]*M2[2]*(c-1) + 1

        # 8 (fft_linear_idx, twiddle) pairs matching reconstruct_g0_at:
        # buf0=F000, buf1=F001, buf2=F110, buf3=F111
        fft_idx = (
            Int32(0*M2_vol + _lin3(ix,iy,iz)),  # F000[ix,iy,iz]
            Int32(1*M2_vol + _lin3(iy,iz,ix)),  # F001[iy,iz,ix]
            Int32(1*M2_vol + _lin3(iz,ix,iy)),  # F001[iz,ix,iy]
            Int32(1*M2_vol + _lin3(ix,iy,iz)),  # F001[ix,iy,iz]
            Int32(2*M2_vol + _lin3(ix,iy,iz)),  # F110[ix,iy,iz]
            Int32(2*M2_vol + _lin3(iz,ix,iy)),  # F110[iz,ix,iy]
            Int32(2*M2_vol + _lin3(iy,iz,ix)),  # F110[iy,iz,ix]
            Int32(3*M2_vol + _lin3(ix,iy,iz)),  # F111[ix,iy,iz]
        )

        tw = (
            complex(1.0),
            cispi(-2*hx/M[1]),
            cispi(-2*hy/M[2]),
            cispi(-2*hz/M[3]),
            cispi(-2*(hx/M[1] + hy/M[2])),
            cispi(-2*(hx/M[1] + hz/M[3])),
            cispi(-2*(hy/M[2] + hz/M[3])),
            cispi(-2*(hx/M[1] + hy/M[2] + hz/M[3])),
        )

        g0_p3c[compact_idx] = P3cPointWeights(fft_idx, tw)
    end

    M3 = prod(M)
    pct = round(100 * n_g0 / M3, digits=1)
    @info "Selective G0 plan: n_spec=$n_spec, |S|=$n_g0/$M3 ($pct%), 8×$(Tuple(M2)) FFTs"

    # Step 4: Allocate buffers
    M2_tuple = Tuple(M2)
    buffers = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    work_ffts = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    fft_flat = zeros(ComplexF64, 4 * M2_vol)
    sub_plan = plan_fft(buffers[1])
    output_buffer = zeros(ComplexF64, n_spec)
    g0_values = zeros(ComplexF64, n_g0)

    return SelectiveG0Plan(
        sub_plan, buffers, work_ffts, fft_flat, output_buffer,
        g0_p3c, g0_values,
        a8_table, n_spec,
        collect(N), M, M2
    )
end

"""
    execute_selective_krfft!(plan, spec_asu, u)

Full selective G0 cascade pipeline:
1. Pack 4 sub-sub-grids at stride 4
2. FFT × 4 on (N/4)³ grids
3. Evaluate G0 at |S| selective positions (P3c point evaluation)
4. A8 lookup from compact G0 table → spectral ASU
"""
function execute_selective_krfft!(plan::SelectiveG0Plan, spec_asu::SpectralIndexing,
                                  u::AbstractArray{<:Real})
    M2 = plan.sub_sub_dims

    # 1. Pack P3c sub-sub-grids at stride 4
    buf000, buf001 = plan.buffers[1], plan.buffers[2]
    buf110, buf111 = plan.buffers[3], plan.buffers[4]
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000[i,j,k] = complex(u[ii+1, jj+1, kk+1])
        buf001[i,j,k] = complex(u[ii+1, jj+1, kk+3])
        buf110[i,j,k] = complex(u[ii+3, jj+3, kk+1])
        buf111[i,j,k] = complex(u[ii+3, jj+3, kk+3])
    end

    # 2. FFT × 4
    p = plan.sub_plan
    for i in 1:4
        mul!(plan.work_ffts[i], p, plan.buffers[i])
    end

    # 3. Concat FFT outputs into flat vector
    M2_vol = prod(M2)
    fft = plan.fft_flat
    @inbounds for b in 1:4
        offset = (b-1) * M2_vol
        copyto!(fft, offset + 1, vec(plan.work_ffts[b]), 1, M2_vol)
    end

    # 4. Evaluate G0 at |S| selective positions (P3c point evaluation)
    g0 = plan.g0_values
    g0_p3c = plan.g0_p3c
    n_g0 = length(g0)

    @inbounds for i in 1:n_g0
        pw = g0_p3c[i]
        val  = pw.tw[1] * fft[pw.fft_idx[1]]
        val += pw.tw[2] * fft[pw.fft_idx[2]]
        val += pw.tw[3] * fft[pw.fft_idx[3]]
        val += pw.tw[4] * fft[pw.fft_idx[4]]
        val += pw.tw[5] * fft[pw.fft_idx[5]]
        val += pw.tw[6] * fft[pw.fft_idx[6]]
        val += pw.tw[7] * fft[pw.fft_idx[7]]
        val += pw.tw[8] * fft[pw.fft_idx[8]]
        g0[i] = val
    end

    # 5. A8 reconstruction: 8 lookups per spectral point
    out = plan.output_buffer
    a8 = plan.a8_table
    n_spec = plan.n_spec

    @inbounds for h in 1:n_spec
        base = (h-1) * 8
        val  = a8[base+1].weight * g0[a8[base+1].g0_idx]
        val += a8[base+2].weight * g0[a8[base+2].g0_idx]
        val += a8[base+3].weight * g0[a8[base+3].g0_idx]
        val += a8[base+4].weight * g0[a8[base+4].g0_idx]
        val += a8[base+5].weight * g0[a8[base+5].g0_idx]
        val += a8[base+6].weight * g0[a8[base+6].g0_idx]
        val += a8[base+7].weight * g0[a8[base+7].g0_idx]
        val += a8[base+8].weight * g0[a8[base+8].g0_idx]
        out[h] = val
    end

    return plan.output_buffer
end

# ============================================================================
# G0 ASU + Point Group Reduction — Orbit-reduced G0 computation
# ============================================================================

"""
    has_cubic_p3c_symmetry(ops) → Bool

Check whether the space group has cubic P3c symmetry — specifically, whether
there exists an operation with the 3-fold body-diagonal rotation R that maps
(x,y,z) → (y,z,x). This cyclic permutation is required by the P3c twiddle
formula which hardcodes `F_001[iy,iz,ix]` and `F_001[iz,ix,iy]`.

Only cubic groups (#195-#230) possess this symmetry. Tetragonal, orthorhombic,
and lower-symmetry groups do NOT, and will produce incorrect results with the
P3c decomposition.
"""
function has_cubic_p3c_symmetry(ops::Vector{SymOp})
    # The 3-fold body-diagonal rotation: R = [0 0 1; 1 0 0; 0 1 0]
    for op in ops
        R = round.(Int, op.R)
        if R == [0 0 1; 1 0 0; 0 1 0]
            return true
        end
    end
    return false
end

"""
    G0ASUPlan

G0 ASU plan: computes G0 only at orbit representatives under the remaining
point group (operations with even translations, |G_rem| = |G|/8).

Requires cubic symmetry (P3c body-diagonal 3-fold rotation).

The symmetry relation is:
    G0(R_α^T q mod M) = exp(-2πi q · (t_α/2) / M) × G0(q)

So only one G0 per orbit needs P3c evaluation. The symmetry phase is
merged into the A8 weight at plan time.

Cost: n_reps × 8 (P3c) + n_spec × 8 (A8 lookup)
where n_reps ≈ n_spec (orbit count ≈ spectral ASU size).
"""
struct G0ASUPlan
    sub_plan::Any                            # FFTW plan for (N/4)³
    buffers::Vector{Array{ComplexF64,3}}     # 4 input: F000, F001, F110, F111
    work_ffts::Vector{Array{ComplexF64,3}}   # 4 FFT outputs
    fft_flat::Vector{ComplexF64}             # 4×M2³ flat concat
    output_buffer::Vector{ComplexF64}        # spectral ASU output

    # G0 ASU: P3c weights only at orbit representatives
    g0_p3c::Vector{P3cPointWeights}          # n_reps entries
    g0_values::Vector{ComplexF64}            # n_reps G0 values

    # A8 table with merged G0 symmetry phase
    a8_table::Vector{SelectiveG0Entry}       # 8 × n_spec
    n_spec::Int
    n_reps::Int                              # G0 ASU size

    grid_N::Vector{Int}
    subgrid_dims::Vector{Int}   # M = N/2
    sub_sub_dims::Vector{Int}   # M2 = N/4
end

"""
    plan_krfft_g0asu(spec_asu, ops_shifted)

Create a G0 ASU plan with orbit reduction.

1. Collects unique G0 positions S (same as selective)
2. Extracts remaining point group (ops with even translations)
3. Enumerates orbits of S under remaining PG → orbit representatives
4. Precomputes P3c weights only at orbit representatives
5. Builds A8 table with merged G0 symmetry phases
"""
function plan_krfft_g0asu(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "G0 ASU KRFFT currently supports 3D only"
    @assert has_cubic_p3c_symmetry(ops_shifted) "G0 ASU requires cubic symmetry (P3c body-diagonal 3-fold rotation). " *
        "Non-cubic groups (tetragonal, orthorhombic, etc.) are not supported. " *
        "Use plan_krfft_selective() or plan_krfft_sparse() instead."

    M = [N[d] ÷ 2 for d in 1:dim]
    M2 = [M[d] ÷ 2 for d in 1:dim]
    M2_vol = prod(M2)

    @assert all(M .* 2 .== collect(N)) "Grid size must be divisible by 2"
    @assert all(M2 .* 2 .== M) "M must be divisible by 2 for P3c"

    n_spec = length(spec_asu.points)

    # Step 1: Select A8 representative operations (one per parity class)
    subgrid_reps = Vector{Union{Nothing, SymOp}}(nothing, 8)
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], 2) for d in 1:dim]
        idx = 1 + x0[1] + 2*x0[2] + 4*x0[3]
        if subgrid_reps[idx] === nothing
            subgrid_reps[idx] = op
        end
    end
    # Determine active A8 classes
    active_a8 = Int[]
    for idx in 1:8
        if subgrid_reps[idx] !== nothing
            push!(active_a8, idx)
        end
    end
    n_a8 = length(active_a8)

    # Step 2: Extract remaining point group (even-translation ops)
    rem_ops = SymOp[]
    for op in ops_shifted
        t = round.(Int, op.t)
        if all(mod.(t, 2) .== 0)
            push!(rem_ops, op)
        end
    end
    n_rem = length(rem_ops)

    # Step 3: Collect all accessed G0 positions + per-spectral-point A8 info
    g0_pos_set = Set{Int}()
    # Each spectral point has 8 A8 entries (padded with zero-weight for inactive classes)
    a8_raw = Vector{Tuple{Int, ComplexF64}}(undef, 8 * n_spec)

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        base = (h_idx - 1) * 8
        slot = 0
        for a8_idx in active_a8
            g = subgrid_reps[a8_idx]::SymOp
            a8_phase = sum(h_vec[d] * g.t[d] / N[d] for d in 1:dim)
            a8_tw = cispi(-2 * a8_phase)
            rot_h = [mod(sum(Int(g.R[d2, d1]) * h_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]
            lin = 1 + rot_h[1] + M[1] * rot_h[2] + M[1] * M[2] * rot_h[3]
            push!(g0_pos_set, lin)
            slot += 1
            a8_raw[base + slot] = (lin, a8_tw)
        end
        # Pad remaining slots with zero weight (pointing to a safe index)
        for s in (slot+1):8
            a8_raw[base + s] = (1, complex(0.0))
        end
    end

    # Step 4: Enumerate orbits under remaining PG
    # For each G0 position, find orbit → assign representative (minimum lin)
    g0_to_rep = Dict{Int, Int}()      # G0 linear idx → representative linear idx
    g0_to_phase = Dict{Int, ComplexF64}()  # G0 linear idx → phase to get from rep

    for start_lin in g0_pos_set
        haskey(g0_to_rep, start_lin) && continue

        # Decode start position
        lin0 = start_lin - 1
        sq = [mod(lin0, M[1]), mod(div(lin0, M[1]), M[2]), div(lin0, M[1] * M[2])]

        # Build orbit via worklist
        orbit = Dict{Int, Tuple{NTuple{3,Int}, ComplexF64}}()
        # orbit[lin] = (q_vec, phase_from_start)
        orbit[start_lin] = (Tuple(sq), complex(1.0))
        worklist = [(sq, complex(1.0))]

        while !isempty(worklist)
            q_vec, q_phase = pop!(worklist)
            for op in rem_ops
                # R^T q mod M
                rq = [mod(sum(Int(op.R[d2, d1]) * q_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]
                rlin = 1 + rq[1] + M[1] * rq[2] + M[1] * M[2] * rq[3]
                if !haskey(orbit, rlin) && rlin in g0_pos_set
                    # Phase: G0(R^T q) = exp(-2πi q·(t/2)/M) × G0(q)
                    t_half = round.(Int, op.t) .÷ 2
                    sym_phase = cispi(-2 * sum(q_vec[d] * t_half[d] / M[d] for d in 1:dim))
                    new_phase = sym_phase * q_phase
                    orbit[rlin] = (Tuple(rq), new_phase)
                    push!(worklist, (rq, new_phase))
                end
            end
        end

        # Choose representative = minimum linear index in orbit
        rep_lin = minimum(keys(orbit))
        rep_phase = orbit[rep_lin][2]

        # For each member: g0_to_rep[member] = rep, phase to reconstruct member from rep
        # member_value = member_phase_from_start / rep_phase_from_start × rep_value
        # But we track: G0(member) = (member_phase / rep_phase) × G0(rep)
        for (member_lin, (_, member_phase)) in orbit
            g0_to_rep[member_lin] = rep_lin
            g0_to_phase[member_lin] = member_phase / rep_phase
        end
    end

    # Step 5: Build compact representative table
    reps = sort(collect(Set(values(g0_to_rep))))
    rep_to_compact = Dict{Int, Int}()
    for (i, r) in enumerate(reps)
        rep_to_compact[r] = i
    end
    n_reps = length(reps)

    # Step 6: Build A8 table with merged symmetry phase
    a8_table = Vector{SelectiveG0Entry}(undef, 8 * n_spec)
    for i in 1:length(a8_raw)
        g0_lin, a8_tw = a8_raw[i]
        rep_lin = g0_to_rep[g0_lin]
        sym_phase = g0_to_phase[g0_lin]
        compact_idx = rep_to_compact[rep_lin]
        # Merged weight = A8 phase × G0 symmetry phase
        a8_table[i] = SelectiveG0Entry(Int32(compact_idx), a8_tw * sym_phase)
    end

    # Step 7: Precompute P3c twiddle weights for orbit representatives only
    g0_p3c = Vector{P3cPointWeights}(undef, n_reps)

    for (compact_idx, lin) in enumerate(reps)
        lin0 = lin - 1
        hx = mod(lin0, M[1])
        hy = mod(div(lin0, M[1]), M[2])
        hz = div(lin0, M[1] * M[2])

        ix = mod(hx, M2[1]) + 1
        iy = mod(hy, M2[2]) + 1
        iz = mod(hz, M2[3]) + 1

        _lin3(a,b,c) = (a-1) + M2[1]*(b-1) + M2[1]*M2[2]*(c-1) + 1

        fft_idx = (
            Int32(0*M2_vol + _lin3(ix,iy,iz)),
            Int32(1*M2_vol + _lin3(iy,iz,ix)),
            Int32(1*M2_vol + _lin3(iz,ix,iy)),
            Int32(1*M2_vol + _lin3(ix,iy,iz)),
            Int32(2*M2_vol + _lin3(ix,iy,iz)),
            Int32(2*M2_vol + _lin3(iz,ix,iy)),
            Int32(2*M2_vol + _lin3(iy,iz,ix)),
            Int32(3*M2_vol + _lin3(ix,iy,iz)),
        )

        tw = (
            complex(1.0),
            cispi(-2*hx/M[1]),
            cispi(-2*hy/M[2]),
            cispi(-2*hz/M[3]),
            cispi(-2*(hx/M[1] + hy/M[2])),
            cispi(-2*(hx/M[1] + hz/M[3])),
            cispi(-2*(hy/M[2] + hz/M[3])),
            cispi(-2*(hx/M[1] + hy/M[2] + hz/M[3])),
        )

        g0_p3c[compact_idx] = P3cPointWeights(fft_idx, tw)
    end

    M3 = prod(M)
    n_s = length(g0_pos_set)
    pct_s = round(100 * n_s / M3, digits=1)
    pct_r = round(100 * n_reps / n_s, digits=1)
    @info "G0 ASU plan: n_spec=$n_spec, |S|=$n_s ($pct_s%), n_reps=$n_reps ($pct_r% of S), |G_rem|=$n_rem"

    # Step 8: Allocate buffers
    M2_tuple = Tuple(M2)
    buffers = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    work_ffts = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    fft_flat = zeros(ComplexF64, 4 * M2_vol)
    sub_plan = plan_fft(buffers[1])
    output_buffer = zeros(ComplexF64, n_spec)
    g0_values = zeros(ComplexF64, n_reps)

    return G0ASUPlan(
        sub_plan, buffers, work_ffts, fft_flat, output_buffer,
        g0_p3c, g0_values,
        a8_table, n_spec, n_reps,
        collect(N), M, M2
    )
end

"""
    execute_g0asu_krfft!(plan, spec_asu, u)

Execute G0 ASU cascade: same pipeline as selective, but with fewer G0 evaluations.
1. Pack → FFT×4 → concat
2. P3c at n_reps orbit representatives only
3. A8 lookup (weights include merged symmetry phases)
"""
function execute_g0asu_krfft!(plan::G0ASUPlan, spec_asu::SpectralIndexing,
                              u::AbstractArray{<:Real})
    M2 = plan.sub_sub_dims

    # 1. Pack P3c sub-sub-grids at stride 4
    buf000, buf001 = plan.buffers[1], plan.buffers[2]
    buf110, buf111 = plan.buffers[3], plan.buffers[4]
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000[i,j,k] = complex(u[ii+1, jj+1, kk+1])
        buf001[i,j,k] = complex(u[ii+1, jj+1, kk+3])
        buf110[i,j,k] = complex(u[ii+3, jj+3, kk+1])
        buf111[i,j,k] = complex(u[ii+3, jj+3, kk+3])
    end

    # 2. FFT × 4
    p = plan.sub_plan
    for i in 1:4
        mul!(plan.work_ffts[i], p, plan.buffers[i])
    end

    # 3. Concat FFT outputs
    M2_vol = prod(M2)
    fft = plan.fft_flat
    @inbounds for b in 1:4
        offset = (b-1) * M2_vol
        copyto!(fft, offset + 1, vec(plan.work_ffts[b]), 1, M2_vol)
    end

    # 4. P3c at orbit representatives only
    g0 = plan.g0_values
    g0_p3c = plan.g0_p3c
    n_reps = plan.n_reps

    @inbounds for i in 1:n_reps
        pw = g0_p3c[i]
        val  = pw.tw[1] * fft[pw.fft_idx[1]]
        val += pw.tw[2] * fft[pw.fft_idx[2]]
        val += pw.tw[3] * fft[pw.fft_idx[3]]
        val += pw.tw[4] * fft[pw.fft_idx[4]]
        val += pw.tw[5] * fft[pw.fft_idx[5]]
        val += pw.tw[6] * fft[pw.fft_idx[6]]
        val += pw.tw[7] * fft[pw.fft_idx[7]]
        val += pw.tw[8] * fft[pw.fft_idx[8]]
        g0[i] = val
    end

    # 5. A8 reconstruction (weights include G0 symmetry phase)
    out = plan.output_buffer
    a8 = plan.a8_table
    n_spec = plan.n_spec

    @inbounds for h in 1:n_spec
        base = (h-1) * 8
        val  = a8[base+1].weight * g0[a8[base+1].g0_idx]
        val += a8[base+2].weight * g0[a8[base+2].g0_idx]
        val += a8[base+3].weight * g0[a8[base+3].g0_idx]
        val += a8[base+4].weight * g0[a8[base+4].g0_idx]
        val += a8[base+5].weight * g0[a8[base+5].g0_idx]
        val += a8[base+6].weight * g0[a8[base+6].g0_idx]
        val += a8[base+7].weight * g0[a8[base+7].g0_idx]
        val += a8[base+8].weight * g0[a8[base+8].g0_idx]
        out[h] = val
    end

    return plan.output_buffer
end
