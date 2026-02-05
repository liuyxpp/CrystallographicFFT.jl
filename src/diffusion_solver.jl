module DiffusionSolver

using LinearAlgebra
using SparseArrays
using ..SymmetryOps
using ..ASU
using ..SpectralIndexing
using ..MatrixQ

export DiffusionSolver, plan_diffusion, step_diffusion!, apply_diffusion_operator!

struct DiffusionSolver{MType, QType}
    real_asu::Vector{ASUPoint}      # Flattened list of real ASU points
    spec_asu::SpectralIndexing
    
    # Linear Operators
    Q::QType                 # Spectral Diffusion Operator (Diagonal/Sparse)
    M::MType                 # Spectral -> Real Transform (Inverse FFT)
    M_inv::MType             # Real -> Spectral Transform (Forward FFT)
    
    # State vectors (internal buffers if needed, or just handle input/output)
    # We'll match Polyorder's in-place style.
end


"""
    calc_transform_matrix(real_pts, spec_asu, ops, N, shift) -> Matrix

Compute matrix M where u_real = M * u_spec.
Row i (real point r_i)
Col j (spectral repr k_j)
M_ij = BasisFunc_j(r_i)

The shift is the "magic shift" vector (in fractional coordinates) used for the real grid.
`ops` must be the symmetry operations used to generate the ASU (likely shifted).
"""
function calc_transform_matrix(real_pts::Vector{ASUPoint}, spec_asu::SpectralIndexing, direct_ops::Vector{SymOp}, N::Tuple, shift::NTuple)
    n_real = length(real_pts)
    n_spec = length(spec_asu.points)
    
    M = zeros(ComplexF64, n_real, n_spec)
    
    # Direct operations (G) provided
    # Reciprocal operations (G*) corresponding to indices in spec_asu
    recip_ops = spec_asu.ops
    
    n_ops = length(direct_ops)
    
    for (j, k_pt) in enumerate(spec_asu.points)
        k_idx = get_k_vector(spec_asu, j) # Integer vector k (frequency)
        
        for (i, r_pt) in enumerate(real_pts)
            # r_idx are integer indices on the shifted grid 0..N-1
            r_idx = r_pt.idx
            
            val = 0.0 + 0.0im
            
            for (iop, op) in enumerate(direct_ops)
                # op has R, t (integers).
                # We want to evaluate plane wave at g^{-1} r.
                # However, cleaner is: Basis = (1/|G|) sum_g exp(i k . (g^-1 r))
                # = (1/|G|) sum_g exp(i (g k) . r)
                # But we must include phase shifts from non-symmorphic t.
                # The correct action of g on k is NOT just rotation.
                # It involves phase e^{-i g k . t_g}.
                # Basis_k(r) = (1/|G|) sum_g exp(-i (g k) . t_g) * exp(i (g k) . r)
                # Let's verify this formula.
                # T_g f(r) = f(g^{-1} r).
                # f(r) must satisfy T_g f = f.
                # T_g e^{i k r} = e^{i k (g^{-1} r)} = e^{i k (R^{-1} r - R^{-1} t)}
                # = e^{-i k R^{-1} t} e^{i (R^{-T} k) . r}.
                # So applying T_g generates a term with wavevector k' = R^{-T} k
                # and phase exp(-i k . R^{-1} t) = exp(-i (R^{-T} k) . t).
                # So yes: sum_g exp(-i (g_dual k) . t_g) * exp(i (g_dual k) . r)
                
                # g_dual is exactly recip_ops[iop].
                R_dual = recip_ops[iop].R
                k_prime = R_dual * k_idx
                
                # Phase correction: -k' . t_g
                phase_shift = 0.0
                for d in 1:length(N)
                    phase_shift -= k_prime[d] * op.t[d] / N[d]
                end
                
                # Main wave phase: k' . r_phys
                # r_phys = (r_idx + shift) / N
                phase_wave = 0.0
                for d in 1:length(N)
                    coordinate = (r_idx[d] + shift[d] * N[d]) # Effective coordinate index? NO. shift is fractional.
                    # shift[d] * N[d] is usually integer? Not necessarily (rational).
                    # But coordinate * k_prime / N
                    # = (r_idx[d] + shift[d]*N[d]) * k_prime[d] / N[d]
                    # = r_idx[d]*k_prime[d]/N[d] + shift[d]*k_prime[d].
                    
                    term1 = r_idx[d] * k_prime[d] / N[d]
                    term2 = shift[d] * k_prime[d]
                    phase_wave += term1 + term2
                end
                
                total_phase = phase_wave + phase_shift
                val += exp(im * 2π * total_phase)
            end
            
            M[i, j] = val / n_ops # Normalization convention P = 1/|G| sum T_g
        end
    end
    
    return M
end


"""
    plan_diffusion(N::Tuple, lattice::AbstractMatrix, sg_num::Int, dim::Int, Δs::Float64)

Create a DiffusionSolver for the given system.
"""
function plan_diffusion(N::Tuple, lattice::AbstractMatrix, sg_num::Int, dim::Int, Δs::Float64)
    # 1. Real Space ASU with Magic Shift
    direct_ops = get_ops(sg_num, dim, N)
    
    # Use find_optimal_shift to get the best shift and the corresponding ops
    # We must access internal ASU function or duplicate logic?
    # `ASU.find_optimal_shift` is exported.
    
    real_shift, shifted_ops = find_optimal_shift(direct_ops, N)
    
    # Calculate real points using the shifted ops
    # The shift is returned by find_optimal_shift
    real_pts = calc_asu(N, shifted_ops)
    
    # 2. Spectral ASU
    # MUST use the SAME shifted_ops to ensure spectral basis is compatible (extinction rules)
    # Note: dim and N are passed for context, but ops define the symmetry.
    spec_asu = calc_spectral_asu(shifted_ops, dim, N)
    
    # 3. Matrix Q
    kernel_func = calc_gradient_term(N, Δs, lattice)
    Q = calc_matrix_q(spec_asu, kernel_func)
    
    # 4. Transform Matrices M and M_inv
    # Now we pass `real_shift` which matches `shifted_ops` t-vectors implicitly?
    # Wait. `calc_transform_matrix` uses `direct_ops` passed to it? 
    # Or should it use `shifted_ops`?
    # `calc_transform_matrix` implementation currently calls `get_ops(sg_num...)`. 
    # This is WRONG. It must use the ops consistent with the ASU.
    # We should update `calc_transform_matrix` signature to take `direct_ops` or `shifted_ops` as well.
    # Actually, `real_shift` vector is needed for coordinate calculation.
    # The symmetry ops t-vector in `shifted_ops` includes the shift effect (t' = t + (I-R)s).
    # If we use `shifted_ops` for M calculation, we should use `real_shift` only for converting indices to physical coords?
    # Let's verify `calc_transform_matrix`.
    
    M = calc_transform_matrix(real_pts, spec_asu, shifted_ops, N, Tuple(real_shift))
    M_inv = pinv(M) 
    
    return DiffusionSolver(real_pts, spec_asu, Q, M, M_inv)
end

"""
    apply_diffusion_operator!(solver::DiffusionSolver, u_real::Vector)

Pure diffusion step (Spectral part only).
"""
function apply_diffusion_operator!(solver::DiffusionSolver, u_real::Vector)
     u_spec = solver.M_inv * u_real
     u_spec = solver.Q * u_spec
     u_temp = solver.M * u_spec
     @. u_real = real(u_temp)
end

"""
    step_diffusion!(solver::DiffusionSolver, u_real::Vector, w_real::Vector, dt::Float64)

Perform one step of Operator Splitting.
Input: u_real (on Real ASU), w_real (potential on Real ASU).
Updates u_real in-place.
"""
function step_diffusion!(solver::DiffusionSolver, u_real::Vector, w_real::Vector, dt::Float64)
    # 1. Real Space Interaction 1
    @. u_real *= exp(-w_real * dt / 2)
    
    # 2. Diffusion
    # Note: solver.Q already contains exp(-k2 * ds).
    # If dt != 1.0 (or whatever unit Q was built with), we need to adjust.
    # Usually plan_diffusion takes Δs, which corresponds to the full step?
    # Or we assume plan_diffusion is called with the time step dt.
    apply_diffusion_operator!(solver, u_real)
    
    # 3. Real Space Interaction 2
    @. u_real *= exp(-w_real * dt / 2)
    
    return u_real
end

end
