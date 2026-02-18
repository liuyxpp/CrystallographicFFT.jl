# ============================================================================
# Execute-time pipelines for Device-Agnostic CFFT
# ============================================================================
#
# Each function orchestrates kernel calls + FFT.
# CPU: optimized scalar loops with @inbounds @simd.
# GPU: KA @kernel dispatch (defined in kernels.jl).
# ============================================================================

using AbstractFFTs: mul!
using KernelAbstractions: get_backend, CPU

# ── Forward pipeline ─────────────────────────────────────────────────────────

"""
    fft_reconstruct!(plan::ForwardPlan)

Combined FFT + reconstruct for forward CFFT.
Assumes subgrid data is already in `plan.input_buffer`.
Writes spectral ASU result to `plan.output_buffer`.
"""
function fft_reconstruct!(plan::ForwardPlan)
    # 1. Out-of-place FFT: input → work
    mul!(plan.work_view, plan.fft_plan, plan.input_view)

    # 2. Reconstruct spectral ASU
    M_vol = prod(plan.M)
    backend = get_backend(plan.output_buffer)
    # Pmmm separable fast path only on CPU (uses scalar loops)
    if backend isa CPU && plan.is_pmmm && plan.n_spec == M_vol && length(plan.phase_factors) == length(plan.M)
        _reconstruct_pmmm!(plan)
    else
        _reconstruct_general!(plan)
    end

    return plan.output_buffer
end

"""Specialized Pmmm separable reconstruction."""
function _reconstruct_pmmm!(plan::ForwardPlan{T, 3}) where T
    M1, M2, M3 = plan.M[1], plan.M[2], plan.M[3]
    Y = plan.work_view
    out = plan.output_buffer
    φ1 = plan.phase_factors[1]
    φ2 = plan.phase_factors[2]
    φ3 = plan.phase_factors[3]

    h_idx = 0
    @inbounds for i in 1:M1
        i_flip = mod(M1 - (i-1), M1) + 1
        φ1i = φ1[i]

        for j in 1:M2
            j_flip = mod(M2 - (j-1), M2) + 1
            φ2j = φ2[j]

            w_pp = one(Complex{T})
            w_mp = φ2j
            w_fp = φ1i
            w_fm = φ1i * φ2j

            for k in 1:M3
                h_idx += 1
                k_flip = mod(M3 - (k-1), M3) + 1
                φ3k = φ3[k]

                y_ppp = Y[i, j, k]
                y_mpp = Y[i_flip, j, k]
                y_pmp = Y[i, j_flip, k]
                y_ppm = Y[i, j, k_flip]
                y_mmp = Y[i_flip, j_flip, k]
                y_mpm = Y[i_flip, j, k_flip]
                y_pmm = Y[i, j_flip, k_flip]
                y_mmm = Y[i_flip, j_flip, k_flip]

                sum_p = w_pp * y_ppp + w_fp * y_mpp +
                        w_mp * y_pmp + w_fm * y_mmp
                sum_m = w_pp * y_ppm + w_fp * y_mpm +
                        w_mp * y_pmm + w_fm * y_mmm

                out[h_idx] = sum_p + φ3k * sum_m
            end
        end
    end
    return out
end

"""General table-based reconstruction."""
function _reconstruct_general!(plan::ForwardPlan{T}) where T
    n_ops = plan.n_ops
    n_spec = plan.n_spec
    buf = plan.work_buffer
    idx = plan.recon_fiber_idx
    w = plan.recon_weight
    out = plan.output_buffer

    backend = get_backend(buf)
    if backend isa CPU
        @inbounds for h in 1:n_spec
            val = zero(Complex{T})
            base = (h - 1) * n_ops
            for g in 1:n_ops
                k = base + g
                val += w[k] * buf[idx[k]]
            end
            out[h] = val
        end
    else
        kernel = reconstruct_general_kernel!(backend)
        kernel(out, buf, idx, w, n_ops; ndrange=n_spec)
    end
    return out
end

# ── Backward pipeline ────────────────────────────────────────────────────────

"""
    execute_backward!(bplan::BackwardPlan, F_spec)

Execute backward transform: spectral ASU → subgrid.
Returns `bplan.f0_buffer`.
"""
function execute_backward!(bplan::BackwardPlan, F_spec::AbstractVector{<:Complex})
    # 1. Inverse reconstruction
    _inv_reconstruct!(bplan, F_spec)

    # 2. IFFT
    mul!(bplan.f0_view, bplan.ifft_plan, bplan.Y_view)

    return bplan.f0_buffer
end

"""SoA inverse reconstruction: F_spec → Y₀(M³)."""
function _inv_reconstruct!(bplan::BackwardPlan{T}, F_spec::AbstractVector{<:Complex}) where T
    d = bplan.d
    M_vol = prod(bplan.M)
    Y = bplan.Y_buffer
    n = bplan.n_spec
    F_work = bplan.F_work
    widx = bplan.inv_work_idx
    w = bplan.inv_weight

    backend = get_backend(Y)
    if backend isa CPU
        # Fill [F; conj(F)]
        @inbounds @simd for i in 1:n
            F_work[i] = F_spec[i]
        end
        @inbounds @simd for i in 1:n
            F_work[n + i] = conj(F_spec[i])
        end

        # SoA gather-multiply-accumulate
        @inbounds for q in 1:M_vol
            base = (q - 1) * d
            val = w[base + 1] * F_work[widx[base + 1]]
            for a in 2:d
                val += w[base + a] * F_work[widx[base + a]]
            end
            Y[q] = val
        end
    else
        # GPU: fused inv_reconstruct — skip F_work, read F_spec directly
        inv_reconstruct_fused_kernel!(backend)(
            Y, F_spec, widx, w, d, Int32(n); ndrange=M_vol)
    end
end

# ============================================================================
# Phase 3 — Centering fold/unfold execute-time functions
# ============================================================================

# ── Centering fold ───────────────────────────────────────────────────────────

"""
    centering_fold!(plan::CenteringFoldPlan, f0)

Fold subgrid f₀ into n_channels alias-folded channels on H³.
"""
function centering_fold!(plan::CenteringFoldPlan, f0::AbstractArray{<:Real, 3})
    if plan.n_channels == 4
        _centering_fold_4ch!(plan, f0)
    else
        _centering_fold_2ch!(plan, f0)
    end
end

"""2-channel fold for F-centering."""
function _centering_fold_2ch!(plan::CenteringFoldPlan{T},
                              f0::AbstractArray{<:Real, 3}) where T
    H1, H2, H3 = plan.H
    buf0 = plan.channel_bufs[1]
    buf1 = plan.channel_bufs[2]
    tw1, tw2, tw3 = plan.twiddle_1d[2]
    signs = plan.sign_table[2]

    backend = get_backend(buf0)
    if backend isa CPU
        @inbounds for iz in 0:H3-1
            tw_z = tw3[iz+1]
            for iy in 0:H2-1
                tw_yz = tw2[iy+1] * tw_z
                for ix in 0:H1-1
                    v000 = f0[ix+1,     iy+1,     iz+1]
                    v100 = f0[ix+H1+1,  iy+1,     iz+1]
                    v010 = f0[ix+1,     iy+H2+1,  iz+1]
                    v110 = f0[ix+H1+1,  iy+H2+1,  iz+1]
                    v001 = f0[ix+1,     iy+1,     iz+H3+1]
                    v101 = f0[ix+H1+1,  iy+1,     iz+H3+1]
                    v011 = f0[ix+1,     iy+H2+1,  iz+H3+1]
                    v111 = f0[ix+H1+1,  iy+H2+1,  iz+H3+1]

                    buf0[ix+1, iy+1, iz+1] = v000+v100+v010+v110+v001+v101+v011+v111

                    val1 = (signs[1]*v000 + signs[2]*v100 +
                            signs[3]*v010 + signs[4]*v110 +
                            signs[5]*v001 + signs[6]*v101 +
                            signs[7]*v011 + signs[8]*v111)
                    buf1[ix+1, iy+1, iz+1] = val1 * (tw1[ix+1] * tw_yz)
                end
            end
        end
    else
        H_vol = H1 * H2 * H3
        s = signs
        centering_fold_2ch_kernel!(backend)(
            buf0, buf1, f0,
            tw1, tw2, tw3,
            s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
            Int32(H1), Int32(H2), Int32(H3);
            ndrange=H_vol)
    end
end

"""4-channel fused WHT fold for I/C/A centering."""
function _centering_fold_4ch!(plan::CenteringFoldPlan{T},
                              f0::AbstractArray{<:Real, 3}) where T
    H1, H2, H3 = plan.H
    buf1 = plan.channel_bufs[1]
    buf2 = plan.channel_bufs[2]
    buf3 = plan.channel_bufs[3]
    buf4 = plan.channel_bufs[4]

    off2 = plan.offsets[2]; idx2 = off2[1] + off2[2]*2 + off2[3]*4 + 1
    off3 = plan.offsets[3]; idx3 = off3[1] + off3[2]*2 + off3[3]*4 + 1
    off4 = plan.offsets[4]; idx4 = off4[1] + off4[2]*2 + off4[3]*4 + 1

    tw1_2, tw2_2, tw3_2 = plan.twiddle_1d[2]
    tw1_3, tw2_3, tw3_3 = plan.twiddle_1d[3]
    tw1_4, tw2_4, tw3_4 = plan.twiddle_1d[4]

    backend = get_backend(buf1)
    if backend isa CPU
        @inbounds for iz in 0:H3-1
            tz2 = tw3_2[iz+1]; tz3 = tw3_3[iz+1]; tz4 = tw3_4[iz+1]
            for iy in 0:H2-1
                tyz2 = tw2_2[iy+1] * tz2
                tyz3 = tw2_3[iy+1] * tz3
                tyz4 = tw2_4[iy+1] * tz4
                for ix in 0:H1-1
                    v000 = f0[ix+1,     iy+1,     iz+1]
                    v100 = f0[ix+H1+1,  iy+1,     iz+1]
                    v010 = f0[ix+1,     iy+H2+1,  iz+1]
                    v110 = f0[ix+H1+1,  iy+H2+1,  iz+1]
                    v001 = f0[ix+1,     iy+1,     iz+H3+1]
                    v101 = f0[ix+H1+1,  iy+1,     iz+H3+1]
                    v011 = f0[ix+1,     iy+H2+1,  iz+H3+1]
                    v111 = f0[ix+H1+1,  iy+H2+1,  iz+H3+1]

                    a0_00 = v000 + v100;  a1_00 = v000 - v100
                    a0_10 = v010 + v110;  a1_10 = v010 - v110
                    a0_01 = v001 + v101;  a1_01 = v001 - v101
                    a0_11 = v011 + v111;  a1_11 = v011 - v111

                    b00_0 = a0_00 + a0_10;  b01_0 = a0_00 - a0_10
                    b10_0 = a1_00 + a1_10;  b11_0 = a1_00 - a1_10
                    b00_1 = a0_01 + a0_11;  b01_1 = a0_01 - a0_11
                    b10_1 = a1_01 + a1_11;  b11_1 = a1_01 - a1_11

                    wht = (b00_0 + b00_1, b10_0 + b10_1,
                           b01_0 + b01_1, b11_0 + b11_1,
                           b00_0 - b00_1, b10_0 - b10_1,
                           b01_0 - b01_1, b11_0 - b11_1)

                    buf1[ix+1, iy+1, iz+1] = wht[1]
                    buf2[ix+1, iy+1, iz+1] = wht[idx2] * (tw1_2[ix+1] * tyz2)
                    buf3[ix+1, iy+1, iz+1] = wht[idx3] * (tw1_3[ix+1] * tyz3)
                    buf4[ix+1, iy+1, iz+1] = wht[idx4] * (tw1_4[ix+1] * tyz4)
                end
            end
        end
    else
        H_vol = H1 * H2 * H3
        centering_fold_4ch_kernel!(backend)(
            buf1, buf2, buf3, buf4, f0,
            tw1_2, tw2_2, tw3_2,
            tw1_3, tw2_3, tw3_3,
            tw1_4, tw2_4, tw3_4,
            Int32(idx2), Int32(idx3), Int32(idx4),
            Int32(H1), Int32(H2), Int32(H3);
            ndrange=H_vol)
    end
end

# ── FFT channels ─────────────────────────────────────────────────────────────

"""Execute FFT on all folded channels.
On CPU: per-channel loops (n_ch × plan_fft). On GPU: single batched CUFFT.
"""
function fft_channels!(plan::CenteringFoldPlan)
    backend = get_backend(plan.batch_buf)
    if backend isa CPU
        @inbounds for c in 1:plan.n_channels
            mul!(plan.channel_fft_out[c], plan.channel_fft_plans[c], plan.channel_bufs[c])
        end
    else
        # GPU: single batched CUFFT over 4D array (H1, H2, H3, n_ch)
        mul!(plan.batch_fft_out, plan.batch_fft_plan, plan.batch_buf)
    end
end

"""Execute IFFT on all folded channels.
On CPU: per-channel loops. On GPU: single batched CUFFT inverse.
"""
function ifft_channels!(plan::CenteringFoldPlan)
    backend = get_backend(plan.batch_buf)
    if backend isa CPU
        @inbounds for c in 1:plan.n_channels
            mul!(plan.channel_bufs[c], plan.channel_ifft_plans[c], plan.channel_fft_out[c])
        end
    else
        # GPU: single batched inverse CUFFT
        mul!(plan.batch_buf, plan.batch_ifft_plan, plan.batch_fft_out)
    end
end

# ── Assemble / Disassemble G₀ ────────────────────────────────────────────────

"""Assemble G₀ from channel FFT outputs at parity positions h = 2k + off."""
function assemble_G0!(G0::AbstractArray{<:Complex, 3}, plan::CenteringFoldPlan)
    H1, H2, H3 = plan.H
    H_vol = H1 * H2 * H3
    M1, M2, M3 = plan.M
    M_vol = M1 * M2 * M3

    backend = get_backend(G0)
    if backend isa CPU
        fill!(G0, zero(eltype(G0)))
        @inbounds for c in 1:plan.n_channels
            off = plan.offsets[c]
            fft_out = plan.channel_fft_out[c]
            for iz in 0:H3-1, iy in 0:H2-1, ix in 0:H1-1
                h1 = 2*ix + off[1]
                h2 = 2*iy + off[2]
                h3 = 2*iz + off[3]
                G0[h1+1, h2+1, h3+1] = fft_out[ix+1, iy+1, iz+1]
            end
        end
    else
        # GPU: fused assemble — single kernel, no fill needed
        assemble_g0_fused_kernel!(backend)(
            G0, plan.batch_fft_out, plan.alive_mask,
            Int32(M1), Int32(M2), Int32(M3),
            Int32(H1), Int32(H2), Int32(H3);
            ndrange=M_vol)
    end
end

"""Disassemble G₀: extract alive-parity entries into channel fft_out buffers."""
function disassemble_G0!(plan::CenteringFoldPlan,
                         G0::AbstractArray{<:Complex, 3})
    H1, H2, H3 = plan.H
    H_vol = H1 * H2 * H3

    backend = get_backend(G0)
    if backend isa CPU
        @inbounds for c in 1:plan.n_channels
            off = plan.offsets[c]
            fft_out = plan.channel_fft_out[c]
            for iz in 0:H3-1, iy in 0:H2-1, ix in 0:H1-1
                h1 = 2*ix + off[1]
                h2 = 2*iy + off[2]
                h3 = 2*iz + off[3]
                fft_out[ix+1, iy+1, iz+1] = G0[h1+1, h2+1, h3+1]
            end
        end
    else
        for c in 1:plan.n_channels
            off = plan.offsets[c]
            fft_out = plan.channel_fft_out[c]
            disassemble_g0_channel_kernel!(backend)(
                fft_out, G0,
                Int32(off[1]), Int32(off[2]), Int32(off[3]),
                Int32(H1), Int32(H2), Int32(H3);
                ndrange=H_vol)
        end
    end
end

# ── Centering unfold ─────────────────────────────────────────────────────────

"""Inverse of centering_fold!: reconstruct f₀(M³) from channels."""
function centering_unfold!(plan::CenteringFoldPlan, f0::AbstractArray{<:Real, 3})
    if plan.n_channels == 4
        _centering_unfold_4ch!(plan, f0)
    elseif plan.n_channels == 2
        _centering_unfold_2ch!(plan, f0)
    end
end

"""2-channel inverse for F-centering."""
function _centering_unfold_2ch!(plan::CenteringFoldPlan{T},
                                f0::AbstractArray{<:Real, 3}) where T
    H1, H2, H3 = plan.H
    buf0 = plan.channel_bufs[1]
    buf1 = plan.channel_bufs[2]
    tw1, tw2, tw3 = plan.twiddle_1d[2]
    signs = plan.sign_table[2]

    backend = get_backend(buf0)
    if backend isa CPU
        @inbounds for iz in 0:H3-1
            tw_z_conj = conj(tw3[iz+1])
            for iy in 0:H2-1
                tw_yz_conj = conj(tw2[iy+1]) * tw_z_conj
                for ix in 0:H1-1
                    tw_conj = conj(tw1[ix+1]) * tw_yz_conj
                    g0 = real(buf0[ix+1, iy+1, iz+1])
                    g1 = real(buf1[ix+1, iy+1, iz+1] * tw_conj)

                    for eps_idx in 1:8
                        s = signs[eps_idx]
                        val = (g0 + s * g1) / T(8)
                        ex = (eps_idx - 1) & 1
                        ey = ((eps_idx - 1) >> 1) & 1
                        ez = ((eps_idx - 1) >> 2) & 1
                        f0[ix+ex*H1+1, iy+ey*H2+1, iz+ez*H3+1] = val
                    end
                end
            end
        end
    else
        H_vol = H1 * H2 * H3
        s = signs
        centering_unfold_2ch_kernel!(backend)(
            f0, buf0, buf1,
            tw1, tw2, tw3,
            s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
            Int32(H1), Int32(H2), Int32(H3);
            ndrange=H_vol)
    end
end

"""4-channel inverse WHT for I/C/A centering (zero-alloc inner loop)."""
function _centering_unfold_4ch!(plan::CenteringFoldPlan{T},
                                f0::AbstractArray{<:Real, 3}) where T
    H1, H2, H3 = plan.H

    # Precompute alive WHT indices as NTuple (stack-allocated)
    off1 = plan.offsets[1]; aidx1 = off1[1] + off1[2]*2 + off1[3]*4 + 1
    off2 = plan.offsets[2]; aidx2 = off2[1] + off2[2]*2 + off2[3]*4 + 1
    off3 = plan.offsets[3]; aidx3 = off3[1] + off3[2]*2 + off3[3]*4 + 1
    off4 = plan.offsets[4]; aidx4 = off4[1] + off4[2]*2 + off4[3]*4 + 1

    # Extract twiddle tables (references, no allocation)
    tw1_1 = plan.twiddle_1d[1]; tw1_2 = plan.twiddle_1d[2]
    tw1_3 = plan.twiddle_1d[3]; tw1_4 = plan.twiddle_1d[4]

    buf1 = plan.channel_bufs[1]; buf2 = plan.channel_bufs[2]
    buf3 = plan.channel_bufs[3]; buf4 = plan.channel_bufs[4]

    backend = get_backend(buf1)
    if backend isa CPU
        @inbounds for iz in 0:H3-1
            tz1 = conj(tw1_1[3][iz+1]); tz2 = conj(tw1_2[3][iz+1])
            tz3 = conj(tw1_3[3][iz+1]); tz4 = conj(tw1_4[3][iz+1])
            for iy in 0:H2-1
                tyz1 = conj(tw1_1[2][iy+1]) * tz1; tyz2 = conj(tw1_2[2][iy+1]) * tz2
                tyz3 = conj(tw1_3[2][iy+1]) * tz3; tyz4 = conj(tw1_4[2][iy+1]) * tz4
                for ix in 0:H1-1
                    tf1 = conj(tw1_1[1][ix+1]) * tyz1
                    tf2 = conj(tw1_2[1][ix+1]) * tyz2
                    tf3 = conj(tw1_3[1][ix+1]) * tyz3
                    tf4 = conj(tw1_4[1][ix+1]) * tyz4

                    g1 = real(buf1[ix+1, iy+1, iz+1] * tf1)
                    g2 = real(buf2[ix+1, iy+1, iz+1] * tf2)
                    g3 = real(buf3[ix+1, iy+1, iz+1] * tf3)
                    g4 = real(buf4[ix+1, iy+1, iz+1] * tf4)

                    wht = ntuple(Val(8)) do slot
                        slot == aidx1 ? g1 : slot == aidx2 ? g2 :
                        slot == aidx3 ? g3 : slot == aidx4 ? g4 : zero(T)
                    end

                    a0_00 = wht[1] + wht[2]; a1_00 = wht[1] - wht[2]
                    a0_10 = wht[3] + wht[4]; a1_10 = wht[3] - wht[4]
                    a0_01 = wht[5] + wht[6]; a1_01 = wht[5] - wht[6]
                    a0_11 = wht[7] + wht[8]; a1_11 = wht[7] - wht[8]

                    b00_0 = a0_00 + a0_10; b01_0 = a0_00 - a0_10
                    b10_0 = a1_00 + a1_10; b11_0 = a1_00 - a1_10
                    b00_1 = a0_01 + a0_11; b01_1 = a0_01 - a0_11
                    b10_1 = a1_01 + a1_11; b11_1 = a1_01 - a1_11

                    inv8 = one(T) / T(8)
                    f0[ix+1,     iy+1,     iz+1]     = (b00_0 + b00_1) * inv8
                    f0[ix+H1+1,  iy+1,     iz+1]     = (b10_0 + b10_1) * inv8
                    f0[ix+1,     iy+H2+1,  iz+1]     = (b01_0 + b01_1) * inv8
                    f0[ix+H1+1,  iy+H2+1,  iz+1]     = (b11_0 + b11_1) * inv8
                    f0[ix+1,     iy+1,     iz+H3+1]  = (b00_0 - b00_1) * inv8
                    f0[ix+H1+1,  iy+1,     iz+H3+1]  = (b10_0 - b10_1) * inv8
                    f0[ix+1,     iy+H2+1,  iz+H3+1]  = (b01_0 - b01_1) * inv8
                    f0[ix+H1+1,  iy+H2+1,  iz+H3+1]  = (b11_0 - b11_1) * inv8
                end
            end
        end
    else
        H_vol = H1 * H2 * H3
        centering_unfold_4ch_kernel!(backend)(
            f0,
            buf1, buf2, buf3, buf4,
            tw1_1[1], tw1_1[2], tw1_1[3],
            tw1_2[1], tw1_2[2], tw1_2[3],
            tw1_3[1], tw1_3[2], tw1_3[3],
            tw1_4[1], tw1_4[2], tw1_4[3],
            Int32(aidx1), Int32(aidx2), Int32(aidx3), Int32(aidx4),
            Int32(H1), Int32(H2), Int32(H3);
            ndrange=H_vol)
    end
end

# ── Centered forward pipeline ────────────────────────────────────────────────

"""
    fft_reconstruct_centered!(plan::CenteredForwardPlan)

Full centered forward pipeline:
  f₀(M³) → centering_fold → n_ch × H³ FFT → assemble G₀ → reconstruct
Assumes f₀ is already in plan.f0_buffer.
"""
function fft_reconstruct_centered!(plan::CenteredForwardPlan)
    fold = plan.fold_plan
    krfft = plan.krfft_plan

    # 1. Centering fold
    centering_fold!(fold, plan.f0_buffer)

    # 2. FFT each channel
    fft_channels!(fold)

    # 3. Assemble G₀ into work_buffer
    assemble_G0!(plan.G0_view, fold)

    # 4. Reconstruct spectral ASU from G₀ (skip FFT step)
    M_vol = prod(krfft.M)
    backend = get_backend(krfft.output_buffer)
    if backend isa CPU && krfft.is_pmmm && krfft.n_spec == M_vol && length(krfft.phase_factors) == length(krfft.M)
        _reconstruct_pmmm!(krfft)
    else
        _reconstruct_general!(krfft)
    end

    return krfft.output_buffer
end

# ── Centered backward pipeline ──────────────────────────────────────────────

"""CSR orbit-based inverse reconstruction: F_spec → G₀(M³)."""
function _inv_recon_orbit!(G0::AbstractArray{<:Complex, 3},
                           F_spec::AbstractVector{<:Complex},
                           bplan::CenteredBackwardPlan)
    offsets = bplan.inv_offsets
    sidx = bplan.inv_spec_idx
    wt = bplan.inv_weight
    reps = bplan.orbit_rep_pos
    G0_reps = bplan.G0_reps
    n_orbits = bplan.n_orbits

    backend = get_backend(G0_reps)
    if backend isa CPU
        # Step 1: CSR gather at orbit reps
        @inbounds for i in 1:n_orbits
            val = zero(eltype(G0_reps))
            for j in offsets[i]:(offsets[i + 1] - Int32(1))
                h = sidx[j]
                if h > 0
                    val += wt[j] * F_spec[h]
                else
                    val += wt[j] * conj(F_spec[-h])
                end
            end
            G0_reps[i] = val
        end

        # Step 2: Orbit expand to all M_vol positions
        G0_flat = vec(G0)
        oid = bplan.orbit_member_oid
        oph = bplan.orbit_member_phase
        @inbounds for q in eachindex(G0_flat)
            G0_flat[q] = oph[q] * G0_reps[oid[q]]
        end
    else
        # Step 1: CSR orbit gather on device
        orbit_gather_kernel!(backend)(
            G0_reps, F_spec, offsets, sidx, wt;
            ndrange=n_orbits)

        # Step 2: Orbit expand on device
        G0_flat = vec(G0)
        oid = bplan.orbit_member_oid
        oph = bplan.orbit_member_phase
        orbit_expand_kernel!(backend)(
            G0_flat, G0_reps, oid, oph;
            ndrange=length(G0_flat))
    end
end

"""
    execute_centered_backward!(bplan::CenteredBackwardPlan, F_spec)

Full centered backward pipeline:
  F_spec → inv_recon_orbit → G₀ → disassemble → IFFT channels → unfold → f₀
"""
function execute_centered_backward!(bplan::CenteredBackwardPlan,
                                    F_spec::AbstractVector{<:Complex})
    G0 = bplan.G0_view
    fold = bplan.fold_plan

    # 1. CSR orbit inv_recon: F_spec → G₀
    _inv_recon_orbit!(G0, F_spec, bplan)

    # 2. Disassemble G₀ → channel FFT outputs
    disassemble_G0!(fold, G0)

    # 3. IFFT each channel
    ifft_channels!(fold)

    # 4. Centering unfold → f₀
    centering_unfold!(fold, bplan.f0_buffer)

    return bplan.f0_buffer
end
