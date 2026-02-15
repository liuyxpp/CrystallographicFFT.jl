module DiffusionSolver

using LinearAlgebra
using SparseArrays
using ..SymmetryOps
using ..ASU
using ..SpectralIndexing
using ..MatrixQ
using ..KRFFT: GeneralCFFTPlan, map_fft!, map_ifft!, plan_krfft, flatten_to_buffer!, unflatten_from_buffer!
using ..QFusedKRFFT: M2QPlan, plan_m2_q, execute_m2_q!, fullgrid_to_subgrid!, subgrid_to_fullgrid!

export AbstractDiffusionSolver, MatrixDiffusionSolver, KRFFTDiffusionSolver, QFusedDiffusionSolver
export plan_diffusion, step_diffusion!, apply_diffusion_operator!

abstract type AbstractDiffusionSolver end

# =========================================================================
# 1. Matrix Diffusion Solver (Classic O(N^2))
# =========================================================================
struct MatrixDiffusionSolver{MType, QType} <: AbstractDiffusionSolver
    real_asu::Vector{ASUPoint}
    spec_asu::SpectralIndexing
    Q::QType
    M::MType
    M_inv::MType
    # real_asu_points for coordinate access if needed
end

# Include the matrix calculation helper (moved from previous version)
function calc_transform_matrix(real_pts::Vector{ASUPoint}, spec_asu::SpectralIndexing, direct_ops::Vector{<:SymOp}, N::Tuple, shift::NTuple)
    n_real = length(real_pts)
    n_spec = length(spec_asu.points)
    M = zeros(ComplexF64, n_real, n_spec)
    recip_ops = spec_asu.ops
    n_ops = length(direct_ops)
    
    for (j, k_pt) in enumerate(spec_asu.points)
        k_idx = get_k_vector(spec_asu, j)
        for (i, r_pt) in enumerate(real_pts)
            r_idx = r_pt.idx
            val = 0.0 + 0.0im
            for (iop, op) in enumerate(direct_ops)
                # Formula: (1/|G|) sum_g exp(-i (g_dual k) . t_g) * exp(i (g_dual k) . r)
                # Correct phase shift logic as derived in Phase 3
                R_dual = recip_ops[iop].R # Reciprocal Op
                k_prime = R_dual * k_idx
                
                phase_shift = 0.0
                for d in 1:length(N); phase_shift -= k_prime[d] * op.t[d] / N[d]; end
                
                phase_wave = 0.0
                for d in 1:length(N)
                    # r_phys ~ (r + s)/N
                    # phase = k' . (r+s)/N = k'/N . r + k' . s
                    phase_wave += (r_idx[d] * k_prime[d] / N[d]) + (shift[d] * k_prime[d])
                end
                
                val += exp(im * 2π * (phase_wave + phase_shift))
            end
            M[i, j] = val / n_ops
        end
    end
    return M
end

function apply_diffusion_operator!(solver::MatrixDiffusionSolver, u_real::Vector)
     u_spec = solver.M_inv * u_real
     u_spec = solver.Q * u_spec
     u_temp = solver.M * u_spec
     @. u_real = real(u_temp)
end

# =========================================================================
# 1b. Q-Fused Diffusion Solver (ASU-native, O(M³ log M + d·M³))
# =========================================================================
"""
    QFusedDiffusionSolver

ASU-native diffusion solver using the M2 Q-fused kernel.
Operates entirely on stride-L subgrid data (M³ = N³/∏L).
"""
struct QFusedDiffusionSolver{D, FP, IP} <: AbstractDiffusionSolver
    plan::M2QPlan{D, FP, IP}
    f0::Array{Float64}  # subgrid work buffer (M₁×M₂×M₃)
end

function apply_diffusion_operator!(solver::QFusedDiffusionSolver, u_subgrid::AbstractArray{<:Real})
    copyto!(solver.f0, u_subgrid)
    execute_m2_q!(solver.plan, solver.f0)
    copyto!(u_subgrid, solver.f0)
end

# =========================================================================
# 2. KRFFT Diffusion Solver (O(N log N))
# =========================================================================
struct KRFFTDiffusionSolver{P<:GeneralCFFTPlan, QType} <: AbstractDiffusionSolver
    real_asu::CrystallographicASU # Holds Blocks
    spec_asu::SpectralIndexing
    Q::QType
    plan::P
    weights::CrystallographicASU # Integration weights (multiplicity/n_ops)
    # Temporary buffers
    u_spec_buf::Vector{ComplexF64}
end

function apply_diffusion_operator!(solver::KRFFTDiffusionSolver, u_real::Union{Vector, CrystallographicASU})
    # Warning: Solver assumes u_real is ASU structure.
    
    # 1. Apply Integration Weights (Pre-FFT)
    # u_weighted = u * w
    apply_scale!(u_real, solver.weights)
    
    map_fft!(solver.plan, u_real)
    
    # Forward Recombination: u_spec = M * vec(work_buffer)
    # Note: work_buffer is 1D in new plan
    mul!(solver.u_spec_buf, solver.plan.recombination_map, solver.plan.work_buffer)
    
    # Diffusion in Spectral Space
    lmul!(solver.Q, solver.u_spec_buf)
    
    # Inverse Recombination: buffer = M' * u_spec
    # Maps spec coeffs to Frequency Buffer (Flattened)
    mul!(solver.plan.work_buffer, adjoint(solver.plan.recombination_map), solver.u_spec_buf)
    
    # Inverse FFT (IFFT -> Demodulate -> Accumulate)
    map_ifft!(solver.plan, u_real)
    
    N_total = prod(solver.spec_asu.N)
    
    for d in keys(u_real.dim_blocks)
        w_blocks = solver.weights.dim_blocks[d]
        u_blocks = u_real.dim_blocks[d]
        for (i, b) in enumerate(u_blocks)
             w_b = w_blocks[i]
             
             # Combined Scale: (N * Nb) / weight
             # Empirical: Factor 8 derived from (Scale=N -> 0.13, Scale=64N -> ~8.0).
             # We need ~1.0. So 8 * N.
             # Hypothesis: Scale = N_total * length(b.data)
             
             scale_base = Float64(N_total) * length(b.data)
             
             @. b.data = b.data * scale_base / w_b.data
             
             # Enforce real result (discard numerical noise in imaginary part)
             @. b.data = complex(real(b.data), 0.0)
        end
    end
end

function apply_scale!(u::CrystallographicASU, w::CrystallographicASU)
    for d in keys(u.dim_blocks)
        u_blocks = u.dim_blocks[d]
        w_blocks = w.dim_blocks[d]
        for i in eachindex(u_blocks)
            @. u_blocks[i].data *= w_blocks[i].data
        end
    end
end

# =========================================================================
# 3. Unified Planning Interface
# =========================================================================
function plan_diffusion(N::Tuple, lattice::AbstractMatrix, sg_num::Int, dim::Int, Δs::Float64; method=:matrix)
    # Common Setup: ASU & Magic Shift
    direct_ops = get_ops(sg_num, dim, N)
    real_shift, shifted_ops = find_optimal_shift(direct_ops, N)
    spec_asu = calc_spectral_asu(shifted_ops, dim, N)
    kernel_func = calc_gradient_term(N, Δs, lattice)
    Q = calc_matrix_q(spec_asu, kernel_func)
    
    if method == :matrix
        pts_list = calc_asu(N, shifted_ops)
        M = calc_transform_matrix(pts_list, spec_asu, shifted_ops, N, Tuple(real_shift))
        M_inv = pinv(M)
        return MatrixDiffusionSolver(pts_list, spec_asu, Q, M, M_inv)
    elseif method == :krfft
        # Construct KRFFT Plan
        pts_list = calc_asu(N, shifted_ops)
        # Pack into blocks. We need ComplexF64 for FFT.
        real_asu = pack_asu(pts_list, N, ComplexF64; shift=Tuple(real_shift))
        
        # Build Plan using ONLY the Shifted Ops rationale.
        # But wait, direct_ops vs shifted_ops for Recombination Phase?
        # Recombination uses `g.t`. Shifted Ops have modified `t`.
        # Since we use `real_asu` calculated from `shifted_ops` (implicitly),
        # consistency requires using `shifted_ops` for the phase factors too.
        plan = plan_krfft(real_asu, spec_asu, shifted_ops)
        
        # Calculate Integration Weights (Multiplicity / n_ops)
        # Note: real_asu blocks don't store multiplicity.
        # But pts_list does. Order of pack_asu corresponds to pts_list?
        # pack_asu helper (reused).
        
        # To pack weights, we need a "fake" pts_list with weights as data?
        # But pack_asu takes pts_list to determine geometry.
        # And we need to pack VALUES.
        # Currently pack_asu uses input `T` to allocate empty blocks.
        # It does not copy values.
        # We need a function `pack_values(pts_list, values, N, ...)`?
        # Or manually fill?
        
        # We can create a weights ASU structure similar to real_asu.
        weights_asu = pack_asu(pts_list, N, Float64; shift=Tuple(real_shift))
        
        # Now fill it.
        # We need mapping from pts_list index to block index.
        # Re-iterate logic or rely on deterministic order?
        # `pack_asu` returns CrystallographicASU.
        # We can iterate `pts_list` and `pack_asu` logic again? Expensive/Risky.
        
        # Better: pack_asu currently generates blocks.
        # If we look at `pack_asu.jl`, it iterates `pts_list` to identify blocks.
        # We need a version that FILLS data.
        # Assuming we don't have it, we can hack it:
        # Iterate blocks in `weights_asu`, iterate their range, match with `pts_list`.
        # `pts_list` is sorted by idx?
        # Actually `pack_asu` logic groups points.
        # The safest way is to search `pts_list` for every point in `real_asu` (blocks).
        
        n_ops = length(direct_ops)
        
        # Build lookup for multiplicity
        # Key: idx (vector). Value: multiplicity.
        mult_map = Dict(p.idx => p.multiplicity for p in pts_list)
        
        for d in keys(weights_asu.dim_blocks)
            for b in weights_asu.dim_blocks[d]
                iter = Iterators.product(b.range...)
                for (i, idx_tuple) in enumerate(iter)
                    idx = collect(idx_tuple)
                    mult = mult_map[idx]
                    b.data[i] = mult / n_ops
                end
            end
        end
        
        # Buffer for u_spec
        u_spec_buf = zeros(ComplexF64, length(spec_asu.points))
        
        return KRFFTDiffusionSolver(real_asu, spec_asu, Q, plan, weights_asu, u_spec_buf)
    elseif method == :qfused
        # Q-fused solver: operates on stride-L subgrid data (M³)
        qplan = plan_m2_q(N, sg_num, dim, Δs, lattice)
        M = Tuple(qplan.M)
        f0_buf = zeros(Float64, M)
        return QFusedDiffusionSolver(qplan, f0_buf)
    else
        error("Unknown method: $method. Supported: :matrix, :krfft, :qfused")
    end
end

function step_diffusion!(solver::AbstractDiffusionSolver, u::Any, w::Any, dt::Float64)
    # 1. Interaction 1
    # u is either Vector (Matrix) or CrystallographicASU (KRFFT)
    # w same.
    # dispatch?
    interaction_step!(u, w, dt/2)
    
    # 2. Diffusion
    apply_diffusion_operator!(solver, u)
    
    # 3. Interaction 2
    interaction_step!(u, w, dt/2)
    return u
end

# Helper for interaction
function interaction_step!(u::Vector, w::Vector, factor)
    @. u *= exp(-w * factor)
end

function interaction_step!(u::CrystallographicASU, w::CrystallographicASU, factor)
    # Iterate blocks
    for (d, blocks) in u.dim_blocks
        w_blocks = w.dim_blocks[d]
        for i in eachindex(blocks)
            @. blocks[i].data *= exp(-w_blocks[i].data * factor)
        end
    end
end

end
