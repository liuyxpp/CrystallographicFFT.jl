# ============================================================================
# KernelAbstractions @kernel functions for Device-Agnostic CFFT
# ============================================================================
#
# Each kernel parallelizes one workitem per logical output element.
# On CPU, existing scalar loops in execute.jl are used (faster due to @simd).
# On GPU, these KA kernels provide full device execution.
# ============================================================================

using KernelAbstractions

# ── General forward reconstruction ───────────────────────────────────────────

@kernel function reconstruct_general_kernel!(out, @Const(buf), @Const(idx),
                                             @Const(w), n_ops)
    h = @index(Global)
    CT = eltype(out)
    val = zero(CT)
    base = (h - 1) * n_ops
    @inbounds for g in 1:n_ops
        k = base + g
        val += w[k] * buf[idx[k]]
    end
    @inbounds out[h] = val
end

# ── General backward (inverse) reconstruction ────────────────────────────────

@kernel function inv_reconstruct_kernel!(Y, @Const(F_work), @Const(widx),
                                         @Const(w), d)
    q = @index(Global)
    CT = eltype(Y)
    base = (q - 1) * d
    val = zero(CT)
    @inbounds for a in 1:d
        k = base + a
        val += w[k] * F_work[widx[k]]
    end
    @inbounds Y[q] = val
end

# ── Fill F_work = [F; conj(F)] (CPU path only after Phase B) ─────────────────

@kernel function fill_fwork_kernel!(F_work, @Const(F_spec), n)
    i = @index(Global)
    @inbounds if i <= n
        F_work[i] = F_spec[i]
        F_work[n + i] = conj(F_spec[i])
    end
end

# ── Fused inverse reconstruction (GPU: no F_work buffer needed) ──────────────
#
#  Reuses existing inv_work_idx layout:
#    widx[k] in 1..n_spec    → F_spec[idx]         (direct)
#    widx[k] in n_spec+1..2n → conj(F_spec[idx-n])  (conjugate)
#
# Eliminates fill_fwork_kernel! + F_work buffer entirely on GPU.

@kernel function inv_reconstruct_fused_kernel!(Y, @Const(F_spec),
                                               @Const(widx), @Const(w),
                                               d, n_spec::Int32)
    q = @index(Global)
    CT = eltype(Y)
    base = (q - 1) * d
    val = zero(CT)
    @inbounds for a in 1:d
        k = base + a
        idx = widx[k]
        if idx <= n_spec
            val += w[k] * F_spec[idx]
        else
            val += w[k] * conj(F_spec[idx - n_spec])
        end
    end
    @inbounds Y[q] = val
end

# ── Assemble G₀ from channel FFT outputs (per-channel, legacy) ───────────────

@kernel function assemble_g0_channel_kernel!(G0, @Const(fft_out),
                                              off1::Int32, off2::Int32, off3::Int32,
                                              H1::Int32, H2::Int32, H3::Int32)
    lin = @index(Global)
    @inbounds begin
        ix = ((lin - 1) % H1)
        iy = (((lin - 1) ÷ H1) % H2)
        iz = ((lin - 1) ÷ (H1 * H2))
        h1 = 2 * ix + off1
        h2 = 2 * iy + off2
        h3 = 2 * iz + off3
        G0[h1 + 1, h2 + 1, h3 + 1] = fft_out[ix + 1, iy + 1, iz + 1]
    end
end

# ── Fused assemble G₀ from batch FFT output (GPU: single kernel over M_vol) ──
#
# Replaces fill!(G0, 0) + n_ch separate assemble_g0_channel_kernel! calls
# with a single kernel that iterates over ALL M_vol positions.
# alive_mask[parity+1] maps 3-bit parity (εx + 2εy + 4εz) to channel index
# (1..n_ch = alive channel, 0 = dead → write zero).

@kernel function assemble_g0_fused_kernel!(G0, @Const(batch_fft_out),
                                            @Const(alive_mask),
                                            M1::Int32, M2::Int32, M3::Int32,
                                            H1::Int32, H2::Int32, H3::Int32)
    lin = @index(Global)
    @inbounds begin
        m1 = ((lin - 1) % M1)
        m2 = (((lin - 1) ÷ M1) % M2)
        m3 = ((lin - 1) ÷ (M1 * M2))

        # 3-bit parity: εx + 2*εy + 4*εz
        parity = (m1 & 1) + ((m2 & 1) << 1) + ((m3 & 1) << 2)
        ch = alive_mask[parity + 1]

        if ch > Int32(0)
            # This parity has an alive channel: read from batch_fft_out
            ix = m1 >> 1  # = (m1 - off) ÷ 2, since off = m1 & 1 for alive
            iy = m2 >> 1
            iz = m3 >> 1
            G0[m1 + 1, m2 + 1, m3 + 1] = batch_fft_out[ix + 1, iy + 1, iz + 1, ch]
        else
            G0[m1 + 1, m2 + 1, m3 + 1] = zero(eltype(G0))
        end
    end
end

# ── Disassemble G₀ to channel fft_out buffers ────────────────────────────────

@kernel function disassemble_g0_channel_kernel!(fft_out, @Const(G0),
                                                 off1::Int32, off2::Int32, off3::Int32,
                                                 H1::Int32, H2::Int32, H3::Int32)
    lin = @index(Global)
    @inbounds begin
        ix = ((lin - 1) % H1)
        iy = (((lin - 1) ÷ H1) % H2)
        iz = ((lin - 1) ÷ (H1 * H2))
        h1 = 2 * ix + off1
        h2 = 2 * iy + off2
        h3 = 2 * iz + off3
        fft_out[ix + 1, iy + 1, iz + 1] = G0[h1 + 1, h2 + 1, h3 + 1]
    end
end

# ── Centering fold 2-channel (F-centering) ───────────────────────────────────

@kernel function centering_fold_2ch_kernel!(buf0, buf1, @Const(f0),
                                             @Const(tw1), @Const(tw2), @Const(tw3),
                                             s1::Int, s2::Int, s3::Int,
                                             s4::Int, s5::Int, s6::Int,
                                             s7::Int, s8::Int,
                                             H1::Int32, H2::Int32, H3::Int32)
    lin = @index(Global)
    @inbounds begin
        ix = (lin - 1) % H1
        iy = ((lin - 1) ÷ H1) % H2
        iz = (lin - 1) ÷ (H1 * H2)

        v000 = f0[ix+1,     iy+1,     iz+1]
        v100 = f0[ix+H1+1,  iy+1,     iz+1]
        v010 = f0[ix+1,     iy+H2+1,  iz+1]
        v110 = f0[ix+H1+1,  iy+H2+1,  iz+1]
        v001 = f0[ix+1,     iy+1,     iz+H3+1]
        v101 = f0[ix+H1+1,  iy+1,     iz+H3+1]
        v011 = f0[ix+1,     iy+H2+1,  iz+H3+1]
        v111 = f0[ix+H1+1,  iy+H2+1,  iz+H3+1]

        buf0[ix+1, iy+1, iz+1] = v000+v100+v010+v110+v001+v101+v011+v111

        val1 = (s1*v000 + s2*v100 + s3*v010 + s4*v110 +
                s5*v001 + s6*v101 + s7*v011 + s8*v111)
        buf1[ix+1, iy+1, iz+1] = val1 * (tw1[ix+1] * tw2[iy+1] * tw3[iz+1])
    end
end

# ── Centering fold 4-channel (I/C/A centering) ──────────────────────────────

@kernel function centering_fold_4ch_kernel!(buf1, buf2, buf3, buf4,
                                             @Const(f0),
                                             @Const(tw1_2), @Const(tw2_2), @Const(tw3_2),
                                             @Const(tw1_3), @Const(tw2_3), @Const(tw3_3),
                                             @Const(tw1_4), @Const(tw2_4), @Const(tw3_4),
                                             idx2::Int32, idx3::Int32, idx4::Int32,
                                             H1::Int32, H2::Int32, H3::Int32)
    lin = @index(Global)
    @inbounds begin
        ix = (lin - 1) % H1
        iy = ((lin - 1) ÷ H1) % H2
        iz = (lin - 1) ÷ (H1 * H2)

        v000 = f0[ix+1,     iy+1,     iz+1]
        v100 = f0[ix+H1+1,  iy+1,     iz+1]
        v010 = f0[ix+1,     iy+H2+1,  iz+1]
        v110 = f0[ix+H1+1,  iy+H2+1,  iz+1]
        v001 = f0[ix+1,     iy+1,     iz+H3+1]
        v101 = f0[ix+H1+1,  iy+1,     iz+H3+1]
        v011 = f0[ix+1,     iy+H2+1,  iz+H3+1]
        v111 = f0[ix+H1+1,  iy+H2+1,  iz+H3+1]

        # 3-stage WHT butterfly
        a0_00 = v000 + v100;  a1_00 = v000 - v100
        a0_10 = v010 + v110;  a1_10 = v010 - v110
        a0_01 = v001 + v101;  a1_01 = v001 - v101
        a0_11 = v011 + v111;  a1_11 = v011 - v111

        b00_0 = a0_00 + a0_10;  b01_0 = a0_00 - a0_10
        b10_0 = a1_00 + a1_10;  b11_0 = a1_00 - a1_10
        b00_1 = a0_01 + a0_11;  b01_1 = a0_01 - a0_11
        b10_1 = a1_01 + a1_11;  b11_1 = a1_01 - a1_11

        wht1 = b00_0 + b00_1; wht2 = b10_0 + b10_1
        wht3 = b01_0 + b01_1; wht4 = b11_0 + b11_1
        wht5 = b00_0 - b00_1; wht6 = b10_0 - b10_1
        wht7 = b01_0 - b01_1; wht8 = b11_0 - b11_1

        # Select alive WHT coefficients by index
        # Channel 1 is always (0,0,0) → idx=1 → wht1
        buf1[ix+1, iy+1, iz+1] = wht1

        # Channels 2-4 use runtime idx with branchless select
        _select_wht(idx, w1, w2, w3, w4, w5, w6, w7, w8) =
            idx == 1 ? w1 : idx == 2 ? w2 : idx == 3 ? w3 : idx == 4 ? w4 :
            idx == 5 ? w5 : idx == 6 ? w6 : idx == 7 ? w7 : w8

        c2 = _select_wht(idx2, wht1, wht2, wht3, wht4, wht5, wht6, wht7, wht8)
        c3 = _select_wht(idx3, wht1, wht2, wht3, wht4, wht5, wht6, wht7, wht8)
        c4 = _select_wht(idx4, wht1, wht2, wht3, wht4, wht5, wht6, wht7, wht8)

        buf2[ix+1, iy+1, iz+1] = c2 * (tw1_2[ix+1] * tw2_2[iy+1] * tw3_2[iz+1])
        buf3[ix+1, iy+1, iz+1] = c3 * (tw1_3[ix+1] * tw2_3[iy+1] * tw3_3[iz+1])
        buf4[ix+1, iy+1, iz+1] = c4 * (tw1_4[ix+1] * tw2_4[iy+1] * tw3_4[iz+1])
    end
end

# ── Centering unfold 2-channel (F-centering) ─────────────────────────────────

@kernel function centering_unfold_2ch_kernel!(f0, @Const(buf0), @Const(buf1),
                                               @Const(tw1), @Const(tw2), @Const(tw3),
                                               s1::Int, s2::Int, s3::Int,
                                               s4::Int, s5::Int, s6::Int,
                                               s7::Int, s8::Int,
                                               H1::Int32, H2::Int32, H3::Int32)
    lin = @index(Global)
    T = eltype(f0)
    @inbounds begin
        ix = (lin - 1) % H1
        iy = ((lin - 1) ÷ H1) % H2
        iz = (lin - 1) ÷ (H1 * H2)

        tw_conj = conj(tw1[ix+1]) * conj(tw2[iy+1]) * conj(tw3[iz+1])
        g0 = real(buf0[ix+1, iy+1, iz+1])
        g1 = real(buf1[ix+1, iy+1, iz+1] * tw_conj)

        inv8 = one(T) / T(8)
        signs = (s1, s2, s3, s4, s5, s6, s7, s8)
        for eps_idx in 1:8
            s = signs[eps_idx]
            val = (g0 + s * g1) * inv8
            ex = (eps_idx - 1) & 1
            ey = ((eps_idx - 1) >> 1) & 1
            ez = ((eps_idx - 1) >> 2) & 1
            f0[ix + ex*H1 + 1, iy + ey*H2 + 1, iz + ez*H3 + 1] = val
        end
    end
end

# ── Centering unfold 4-channel (I/C/A centering) ─────────────────────────────

@kernel function centering_unfold_4ch_kernel!(f0,
                                               @Const(buf1), @Const(buf2),
                                               @Const(buf3), @Const(buf4),
                                               @Const(tw1_1), @Const(tw2_1), @Const(tw3_1),
                                               @Const(tw1_2), @Const(tw2_2), @Const(tw3_2),
                                               @Const(tw1_3), @Const(tw2_3), @Const(tw3_3),
                                               @Const(tw1_4), @Const(tw2_4), @Const(tw3_4),
                                               aidx1::Int32, aidx2::Int32,
                                               aidx3::Int32, aidx4::Int32,
                                               H1::Int32, H2::Int32, H3::Int32)
    lin = @index(Global)
    T = eltype(f0)
    @inbounds begin
        ix = (lin - 1) % H1
        iy = ((lin - 1) ÷ H1) % H2
        iz = (lin - 1) ÷ (H1 * H2)

        # Undo twiddle for each channel
        tf1 = conj(tw1_1[ix+1]) * conj(tw2_1[iy+1]) * conj(tw3_1[iz+1])
        tf2 = conj(tw1_2[ix+1]) * conj(tw2_2[iy+1]) * conj(tw3_2[iz+1])
        tf3 = conj(tw1_3[ix+1]) * conj(tw2_3[iy+1]) * conj(tw3_3[iz+1])
        tf4 = conj(tw1_4[ix+1]) * conj(tw2_4[iy+1]) * conj(tw3_4[iz+1])

        g1 = real(buf1[ix+1, iy+1, iz+1] * tf1)
        g2 = real(buf2[ix+1, iy+1, iz+1] * tf2)
        g3 = real(buf3[ix+1, iy+1, iz+1] * tf3)
        g4 = real(buf4[ix+1, iy+1, iz+1] * tf4)

        # Scatter into WHT slots
        _sel(slot, a1, a2, a3, a4, i1, i2, i3, i4, Z) =
            slot == i1 ? a1 : slot == i2 ? a2 : slot == i3 ? a3 : slot == i4 ? a4 : Z

        z = zero(T)
        w1 = _sel(Int32(1), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)
        w2 = _sel(Int32(2), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)
        w3 = _sel(Int32(3), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)
        w4 = _sel(Int32(4), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)
        w5 = _sel(Int32(5), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)
        w6 = _sel(Int32(6), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)
        w7 = _sel(Int32(7), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)
        w8 = _sel(Int32(8), g1, g2, g3, g4, aidx1, aidx2, aidx3, aidx4, z)

        # Inverse WHT butterfly
        a0_00 = w1 + w2; a1_00 = w1 - w2
        a0_10 = w3 + w4; a1_10 = w3 - w4
        a0_01 = w5 + w6; a1_01 = w5 - w6
        a0_11 = w7 + w8; a1_11 = w7 - w8

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

# ── Orbit gather (CSR) ───────────────────────────────────────────────────────

@kernel function orbit_gather_kernel!(G0_reps, @Const(F_spec),
                                       @Const(offsets), @Const(sidx), @Const(wt))
    i = @index(Global)
    CT = eltype(G0_reps)
    val = zero(CT)
    @inbounds for j in offsets[i]:(offsets[i + Int32(1)] - Int32(1))
        h = sidx[j]
        if h > Int32(0)
            val += wt[j] * F_spec[h]
        else
            val += wt[j] * conj(F_spec[-h])
        end
    end
    @inbounds G0_reps[i] = val
end

# ── Orbit expand ─────────────────────────────────────────────────────────────

@kernel function orbit_expand_kernel!(G0_flat, @Const(G0_reps),
                                       @Const(oid), @Const(oph))
    q = @index(Global)
    @inbounds G0_flat[q] = oph[q] * G0_reps[oid[q]]
end

# ── Buffer copy kernels ─────────────────────────────────────────────────────

@kernel function copy_real_to_complex_kernel!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function copy_complex_to_real_kernel!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = real(src[i])
end

# ── rfft-aware forward reconstruction ────────────────────────────────────────
# Sign-encoded indices: positive = direct, negative = conjugate

@kernel function reconstruct_rfft_kernel!(out, @Const(buf), @Const(idx),
                                           @Const(w), n_ops)
    h = @index(Global)
    CT = eltype(out)
    val = zero(CT)
    base = (h - 1) * n_ops
    @inbounds for g in 1:n_ops
        k = base + g
        i = idx[k]
        if i > Int32(0)
            val += w[k] * buf[i]
        else
            val += w[k] * conj(buf[-i])
        end
    end
    @inbounds out[h] = val
end

# ── rfft-aware inverse reconstruction ────────────────────────────────────────
# Sign-encoded: positive = F_spec[idx], negative = conj(F_spec[-idx])

@kernel function inv_reconstruct_rfft_kernel!(Y, @Const(F_spec),
                                               @Const(widx), @Const(w), d)
    q = @index(Global)
    CT = eltype(Y)
    base = (q - 1) * d
    val = zero(CT)
    @inbounds for a in 1:d
        k = base + a
        i = widx[k]
        if i > Int32(0)
            val += w[k] * F_spec[i]
        else
            val += w[k] * conj(F_spec[-i])
        end
    end
    @inbounds Y[q] = val
end
