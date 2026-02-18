# ============================================================================
# Execute-time pipelines for Device-Agnostic CFFT
# ============================================================================
#
# Each function orchestrates kernel calls + FFT.
# For Phase 1-2 (CPU path), we use direct loops instead of KA @kernel
# to match the exact semantics of the archived code.
# KA kernels will be integrated once we verify correctness.
# ============================================================================

using AbstractFFTs: mul!

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
    if plan.is_pmmm && plan.n_spec == M_vol && length(plan.phase_factors) == length(plan.M)
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

    @inbounds for h in 1:n_spec
        val = zero(Complex{T})
        base = (h - 1) * n_ops
        for g in 1:n_ops
            k = base + g
            val += w[k] * buf[idx[k]]
        end
        out[h] = val
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
end
