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

    # Select representative operations for 8 A8 subgrids
    subgrid_reps = Vector{SymOp}(undef, n_a8)
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], 2) for d in 1:dim]
        idx = 1 + x0[1] + 2*x0[2] + 4*x0[3]
        subgrid_reps[idx] = op
    end

    # Build A8 reconstruction table: 8 entries per spectral point
    a8_table = Matrix{P3cReconEntry}(undef, n_a8, n_spec)
    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        for (g_idx, g) in enumerate(subgrid_reps)
            # A8 phase
            a8_phase = sum(h_vec[d] * g.t[d] / N[d] for d in 1:dim)
            a8_tw = cispi(-2 * a8_phase)

            # Rotated frequency into G0: R_g^T h mod M
            rot_h = [mod(sum(Int(g.R[d2, d1]) * h_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]

            # Linear index into G0 cache (1-based, column-major)
            lin = 1 + rot_h[1] + M[1] * rot_h[2] + M[1] * M[2] * rot_h[3]
            a8_table[g_idx, h_idx] = P3cReconEntry(Int32(lin), a8_tw)
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
