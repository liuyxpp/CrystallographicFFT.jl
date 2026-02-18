# ============================================================================
# Plan-time logic for Device-Agnostic CFFT (CPU-only)
# ============================================================================
#
# All functions run on the CPU during plan creation.
# Computed tables/buffers are transferred to the target device at the end.
# ============================================================================

using KernelAbstractions
using AbstractFFTs
using FFTW
using LinearAlgebra
using LinearAlgebra: LAPACK

using .SymmetryOps: SymOp, CenteringType, CentP, CentI, CentF, CentC, CentA, detect_centering_type
using .SpectralIndexing: get_k_vector
const SpecASU = SpectralIndexing.SpectralIndexing  # type alias to avoid module/type name collision

# ── Device transfer helper ───────────────────────────────────────────────────

"""
    _to_device(backend, data)

Transfer array to target device.  No-op on CPU.
"""
function _to_device(backend, data::AbstractArray{T}) where T
    if backend isa CPU
        return data
    end
    dev = KernelAbstractions.allocate(backend, T, size(data)...)
    copyto!(dev, data)
    return dev
end

# ── auto_L — stride factor computation ───────────────────────────────────────

"""
    auto_L(ops_shifted) -> Vector{Int}

Determine optimal Cooley-Tukey stride factor L from shifted symmetry operations.
L_d = 2 if any shifted operation has odd t_d, otherwise L_d = 1.
"""
function auto_L(ops_shifted::Vector{<:SymOp})
    dim = length(ops_shifted[1].t)

    L_max = ones(Int, dim)
    for op in ops_shifted
        t = round.(Int, op.t)
        for d in 1:dim
            if mod(t[d], 2) == 1
                L_max[d] = 2
            end
        end
    end

    n_subgrids = prod(L_max)
    n_subgrids <= 1 && return L_max

    # Count reachable subgrids
    reachable = Set{Vector{Int}}()
    for op in ops_shifted
        t = round.(Int, op.t)
        push!(reachable, [mod(t[d], L_max[d]) for d in 1:dim])
    end
    length(reachable) == n_subgrids && return L_max

    # Reduce L to maximize speedup with all subgrids reachable
    best_L = ones(Int, dim)
    for mask in 1:(2^dim - 1)
        L_try = ones(Int, dim)
        for d in 1:dim
            if (mask >> (d-1)) & 1 == 1 && L_max[d] == 2
                L_try[d] = 2
            end
        end
        n_sub = prod(L_try)
        reach = Set{Vector{Int}}()
        for op in ops_shifted
            t = round.(Int, op.t)
            push!(reach, [mod(t[d], L_try[d]) for d in 1:dim])
        end
        if length(reach) == n_sub && n_sub > prod(best_L)
            best_L = L_try
        end
    end
    return best_L
end

# ── _select_rep_ops — representative operation selection ─────────────────────

"""
    _select_rep_ops(ops_shifted, L, N, dim) -> Vector{SymOp}

Select one representative symmetry operation per subgrid coset.
Prefers diagonal R with simple translations for the Pmmm fast path.
"""
function _select_rep_ops(ops_shifted::Vector{<:SymOp}, L, N, dim)
    subgrid_reps = Dict{NTuple{3,Int}, eltype(ops_shifted)}()
    subgrid_quality = Dict{NTuple{3,Int}, Int}()

    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = (mod(t[1], L[1]), mod(t[2], L[2]), mod(t[3], L[3]))
        is_diag = all(op.R[i,j] == 0 for i in 1:dim for j in 1:dim if i != j)
        simple_t = all(mod(t[d], N[d]) ∈ (0, N[d]-1) for d in 1:dim)
        quality = is_diag ? (simple_t ? 2 : 1) : 0
        if !haskey(subgrid_reps, x0) || quality > subgrid_quality[x0]
            subgrid_reps[x0] = op
            subgrid_quality[x0] = quality
        end
    end

    d = prod(L)
    rep_ops = Vector{eltype(ops_shifted)}(undef, d)
    sub_idx = 0
    for x0 in Iterators.product([0:L[dd]-1 for dd in 1:dim]...)
        sub_idx += 1
        x0_tup = (x0[1], x0[2], x0[3])
        if haskey(subgrid_reps, x0_tup)
            rep_ops[sub_idx] = subgrid_reps[x0_tup]
        else
            error("Subgrid x₀=$x0_tup not reachable. auto_L should have prevented this.")
        end
    end
    return rep_ops
end

# ── Pmmm detection ───────────────────────────────────────────────────────────

"""Check if representative ops form a Pmmm-like separable pattern."""
function _is_pmmm_pattern(rep_ops::Vector{<:SymOp}, L, N, dim)
    d = prod(L)
    all_diagonal = all(op -> all(op.R[i,j] == 0 for i in 1:dim for j in 1:dim if i != j), rep_ops)
    simple_translations = all(op -> all(
        round(Int, op.t[dd]) ∈ (0, -1) || mod(round(Int, op.t[dd]), N[dd]) ∈ (0, N[dd]-1)
        for dd in 1:dim
    ), rep_ops)
    return all_diagonal && simple_translations && d == 2^dim
end

# ── SoA reconstruction table building (forward) ─────────────────────────────

"""
    _build_recon_soa(spec_asu, rep_ops, M_sub, N, dim)

Build SoA reconstruction table: (recon_fiber_idx, recon_weight).
Each is a flat vector of length n_ops × n_spec.
"""
function _build_recon_soa(spec_asu::SpecASU, rep_ops::Vector{<:SymOp},
                          M_sub::Vector{Int}, N, dim)
    n_spec = length(spec_asu.points)
    n_ops = length(rep_ops)
    total = n_ops * n_spec

    recon_fiber_idx = Vector{Int32}(undef, total)
    recon_weight = Vector{ComplexF64}(undef, total)

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        for (g_idx, g) in enumerate(rep_ops)
            # Phase: exp(-2πi h·t_g/N)
            phase_val = 0.0
            for d in 1:dim
                phase_val += h_vec[d] * g.t[d] / N[d]
            end
            weight = exp(-im * 2π * phase_val)

            # Rotated frequency: R_g^T h mod M
            lin_idx = 1
            stride = 1
            for d in 1:dim
                rot_h_d = mod(sum(g.R[d2, d] * h_vec[d2] for d2 in 1:dim), M_sub[d])
                lin_idx += rot_h_d * stride
                stride *= M_sub[d]
            end

            k = (h_idx - 1) * n_ops + g_idx
            recon_fiber_idx[k] = Int32(lin_idx)
            recon_weight[k] = weight
        end
    end

    return recon_fiber_idx, recon_weight
end

# ── Phase factors for Pmmm fast path ─────────────────────────────────────────

function _build_phase_factors(::Type{T}, M_sub, N, dim) where T
    CT = Complex{T}
    phase_factors = Vector{Vector{CT}}(undef, dim)
    for d in 1:dim
        phase_factors[d] = [CT(cispi(2 * h / N[d])) for h in 0:M_sub[d]-1]
    end
    return phase_factors
end

# ── plan_forward ─────────────────────────────────────────────────────────────

"""
    plan_forward(::Type{T}, spec_asu, ops_shifted; backend=CPU()) -> ForwardPlan

Construct a device-agnostic forward KRFFT plan.
"""
function plan_forward(::Type{T}, spec_asu::SpecASU,
                      ops_shifted::Vector{<:SymOp};
                      backend=CPU()) where {T<:AbstractFloat}
    CT = Complex{T}
    N = spec_asu.N
    dim = length(N)

    # 1. Auto L and M
    L_vec = auto_L(ops_shifted)
    M_sub = [N[d] ÷ L_vec[d] for d in 1:dim]
    M_vol = prod(M_sub)
    n_spec = length(spec_asu.points)

    # 2. Representative ops
    rep_ops = _select_rep_ops(ops_shifted, L_vec, N, dim)
    n_ops = prod(L_vec)

    # 3. SoA reconstruction table (CPU)
    fiber_idx_cpu, weight_cpu = _build_recon_soa(spec_asu, rep_ops, M_sub, N, dim)
    weight_cpu_T = CT.(weight_cpu)  # Convert to target precision

    # 4. Buffers (CPU first, then to device)
    input_buf_cpu = zeros(CT, M_vol)
    work_buf_cpu  = zeros(CT, M_vol)
    out_buf_cpu   = zeros(CT, n_spec)

    # 5. Transfer to device (before creating FFT plan)
    input_buf = _to_device(backend, input_buf_cpu)
    work_buf  = _to_device(backend, work_buf_cpu)
    out_buf   = _to_device(backend, out_buf_cpu)
    fiber_idx = _to_device(backend, fiber_idx_cpu)
    weight_dev = _to_device(backend, weight_cpu_T)

    # 6. FFT plan on device array (CUFFT auto-dispatches for GPU)
    M_tup = NTuple{dim, Int}(M_sub)
    input_view = reshape(input_buf, M_tup)
    work_view  = reshape(work_buf, M_tup)
    fft_plan = plan_fft(input_view)

    # 7. Pmmm detection and phase factors
    is_pmmm = _is_pmmm_pattern(rep_ops, L_vec, N, dim)
    VA = typeof(out_buf)  # Get concrete device array type
    if is_pmmm
        phase_cpu = _build_phase_factors(T, M_sub, N, dim)
        phase_factors = VA[_to_device(backend, p) for p in phase_cpu]
    else
        phase_factors = VA[]
    end

    N_tup = NTuple{dim, Int}(N)
    L_tup = NTuple{dim, Int}(L_vec)

    return ForwardPlan(
        fft_plan, input_buf, work_buf, out_buf,
        input_view, work_view,
        fiber_idx, weight_dev, n_ops, n_spec,
        is_pmmm, phase_factors,
        M_tup, N_tup, L_tup
    )
end

# ── Spectral reverse lookup (for backward plan) ─────────────────────────────

"""
Build h → (spec_idx, phase, conj_flag) mapping for all full-grid frequencies.
"""
function _build_spectral_reverse_lookup(spec_asu::SpecASU,
                                        ops_shifted::Vector{<:SymOp},
                                        N_vec::Vector{Int}, dim::Int)
    h_to_spec = Dict{NTuple{3,Int}, Tuple{Int, ComplexF64, Bool}}()
    sizehint!(h_to_spec, prod(N_vec))

    n_spec = length(spec_asu.points)
    n_ops = length(ops_shifted)

    # Pre-extract R and t
    R_flat = Array{Int}(undef, dim, dim, n_ops)
    t_flat = Vector{NTuple{3,Int}}(undef, n_ops)
    for (i, op) in enumerate(ops_shifted)
        for dd in 1:dim, dd2 in 1:dim
            R_flat[dd2, dd, i] = Int(op.R[dd2, dd])
        end
        t_flat[i] = (Int(op.t[1]), Int(op.t[2]), Int(op.t[3]))
    end

    h_buf = zeros(Int, dim)
    N1, N2, N3 = N_vec[1], N_vec[2], N_vec[3]

    for (spec_idx, pt) in enumerate(spec_asu.points)
        h_asu = pt.idx
        h1, h2, h3 = h_asu[1], h_asu[2], h_asu[3]

        # Identity
        h_key = (h1, h2, h3)
        if !haskey(h_to_spec, h_key)
            h_to_spec[h_key] = (spec_idx, complex(1.0), false)
        end

        # All ops
        for oi in 1:n_ops
            t_op = t_flat[oi]
            for dd in 1:dim
                s = 0
                for dd2 in 1:dim
                    s += R_flat[dd2, dd, oi] * h_asu[dd2]
                end
                h_buf[dd] = mod(s, N_vec[dd])
            end
            h_rot_key = (h_buf[1], h_buf[2], h_buf[3])

            if !haskey(h_to_spec, h_rot_key)
                phase_val = h1 * t_op[1] / N1 + h2 * t_op[2] / N2 + h3 * t_op[3] / N3
                h_to_spec[h_rot_key] = (spec_idx, cispi(2 * phase_val), false)
            end

            # Hermitian
            h_neg_key = (mod(-h_buf[1], N1), mod(-h_buf[2], N2), mod(-h_buf[3], N3))
            if !haskey(h_to_spec, h_neg_key)
                phase_val = h1 * t_op[1] / N1 + h2 * t_op[2] / N2 + h3 * t_op[3] / N3
                h_to_spec[h_neg_key] = (spec_idx, cispi(-2 * phase_val), true)
            end
        end

        # Hermitian of ASU point itself
        h_neg_asu = (mod(-h1, N1), mod(-h2, N2), mod(-h3, N3))
        if !haskey(h_to_spec, h_neg_asu)
            h_to_spec[h_neg_asu] = (spec_idx, complex(1.0), true)
        end
    end

    return h_to_spec
end

# ── Inverse reconstruction table building ───────────────────────────────────

"""Build inv_recon table via butterfly matrix B(q) inversion."""
function _build_inv_recon_table!(inv_spec_idx::Vector{Int32},
                                 inv_weight_cpu::Vector{ComplexF64},
                                 inv_conj_flag::Vector{Bool},
                                 rep_ops::Vector{<:SymOp},
                                 alphas::Vector{NTuple{3,Int}},
                                 h_to_spec::Dict{NTuple{3,Int}, Tuple{Int, ComplexF64, Bool}},
                                 M_sub::Vector{Int}, N, d::Int, dim::Int)
    M_vol = prod(M_sub)
    B_matrix = zeros(ComplexF64, d, d)
    h_vecs = [zeros(Int, dim) for _ in 1:d]
    ipiv = Vector{Int64}(undef, d)
    rhs = zeros(ComplexF64, d, 1)
    rhs[1] = 1.0
    rhs_work = similar(rhs)
    q_vec = zeros(Int, dim)

    rep_translations = NTuple{3,Int}[(Int(g.t[1]), Int(g.t[2]), Int(g.t[3])) for g in rep_ops]
    N1, N2, N3 = N[1], N[2], N[3]
    M_sub_tup = Tuple(M_sub)
    ci = CartesianIndices(M_sub_tup)
    li = LinearIndices(M_sub_tup)

    for q_cart in ci
        for dd in 1:dim
            q_vec[dd] = q_cart[dd] - 1
        end
        q_lin = li[q_cart]

        # Full-grid frequencies in this fiber
        for a in 1:d
            α = alphas[a]
            for dd in 1:dim
                h_vecs[a][dd] = q_vec[dd] + M_sub[dd] * α[dd]
            end
        end

        # Build butterfly matrix
        fill!(B_matrix, zero(ComplexF64))
        for a in 1:d
            h = h_vecs[a]
            for b in 1:d
                t_b = rep_translations[b]
                phase_val = h[1] * t_b[1] / N1 + h[2] * t_b[2] / N2 + h[3] * t_b[3] / N3
                B_matrix[a, b] = cispi(-2 * phase_val)
            end
        end

        # Solve B^T · x = e₁
        copyto!(rhs_work, rhs)
        LAPACK.getrf!(B_matrix, ipiv)
        LAPACK.getrs!('T', B_matrix, ipiv, rhs_work)

        # Map to spectral ASU
        for a in 1:d
            h = h_vecs[a]
            h_key = (mod(h[1], N1), mod(h[2], N2), mod(h[3], N3))
            k = (q_lin - 1) * d + a

            if haskey(h_to_spec, h_key)
                spec_idx, sym_phase, conj_f = h_to_spec[h_key]
                inv_spec_idx[k] = Int32(spec_idx)
                inv_weight_cpu[k] = rhs_work[a] * sym_phase
                inv_conj_flag[k] = conj_f
            else
                inv_spec_idx[k] = Int32(1)
                inv_weight_cpu[k] = zero(ComplexF64)
                inv_conj_flag[k] = false
            end
        end
    end
end

"""Convert AoS → SoA: conj entries index into n_spec + spec_idx."""
function _build_soa_work_idx(inv_spec_idx::Vector{Int32},
                              inv_conj_flag::Vector{Bool},
                              n_spec::Int)
    soa_len = length(inv_spec_idx)
    inv_work_idx = Vector{Int32}(undef, soa_len)
    for k in 1:soa_len
        inv_work_idx[k] = inv_conj_flag[k] ?
            Int32(n_spec + inv_spec_idx[k]) : inv_spec_idx[k]
    end
    return inv_work_idx
end

# ── plan_backward ────────────────────────────────────────────────────────────

"""
    plan_backward(::Type{T}, spec_asu, ops_shifted; backend=CPU()) -> BackwardPlan

Construct a device-agnostic backward KRFFT plan.
"""
function plan_backward(::Type{T}, spec_asu::SpecASU,
                       ops_shifted::Vector{<:SymOp};
                       backend=CPU()) where {T<:AbstractFloat}
    CT = Complex{T}
    N = spec_asu.N
    dim = length(N)
    N_vec = collect(N)

    # 1. L and M
    L_vec = auto_L(ops_shifted)
    M_sub = [N[d] ÷ L_vec[d] for d in 1:dim]
    M_vol = prod(M_sub)
    n_spec = length(spec_asu.points)
    d = prod(L_vec)

    # 2. Representative ops
    rep_ops = _select_rep_ops(ops_shifted, L_vec, N, dim)

    # 3. Spectral reverse lookup
    h_to_spec = _build_spectral_reverse_lookup(spec_asu, ops_shifted, N_vec, dim)

    # 4. Inverse reconstruction table
    soa_len = d * M_vol
    inv_spec_idx_cpu = Vector{Int32}(undef, soa_len)
    inv_weight_cpu = Vector{ComplexF64}(undef, soa_len)
    inv_conj_flag = Vector{Bool}(undef, soa_len)

    alphas = Vector{NTuple{3,Int}}(undef, d)
    a_idx = 0
    for x0 in Iterators.product([0:L_vec[dd]-1 for dd in 1:dim]...)
        a_idx += 1
        alphas[a_idx] = (x0[1], x0[2], x0[3])
    end
    _build_inv_recon_table!(inv_spec_idx_cpu, inv_weight_cpu, inv_conj_flag,
                            rep_ops, alphas, h_to_spec, M_sub, N, d, dim)

    # 5. SoA work indices
    inv_work_idx_cpu = _build_soa_work_idx(inv_spec_idx_cpu, inv_conj_flag, n_spec)
    inv_weight_T = CT.(inv_weight_cpu)

    # 6. Buffers
    F_work_cpu = zeros(CT, 2 * n_spec)
    Y_buf_cpu = zeros(CT, M_vol)
    f0_buf_cpu = zeros(CT, M_vol)

    # 7. Transfer to device (before creating IFFT plan)
    inv_work_idx = _to_device(backend, inv_work_idx_cpu)
    inv_weight_dev = _to_device(backend, inv_weight_T)
    F_work = _to_device(backend, F_work_cpu)
    Y_buf = _to_device(backend, Y_buf_cpu)
    f0_buf = _to_device(backend, f0_buf_cpu)

    M_tup = NTuple{dim, Int}(M_sub)
    Y_view = reshape(Y_buf, M_tup)
    f0_view = reshape(f0_buf, M_tup)

    # 8. IFFT plan on device array (CUFFT auto-dispatches for GPU)
    ifft_plan = plan_ifft(Y_view)

    # 9. Pmmm detection
    is_sep = _is_pmmm_pattern(rep_ops, L_vec, N, dim)
    VA = typeof(Y_buf)  # Get concrete device array type
    if is_sep
        inv_phase_factors = Vector{VA}(undef, dim)
        for dd in 1:dim
            pf_cpu = [CT(cispi(-2 * q / N[dd])) for q in 0:M_sub[dd]-1]
            inv_phase_factors[dd] = _to_device(backend, pf_cpu)
        end
    else
        inv_phase_factors = VA[]
    end

    N_tup = NTuple{dim, Int}(N)
    L_tup = NTuple{dim, Int}(L_vec)

    return BackwardPlan(
        ifft_plan, Y_buf, f0_buf, Y_view, f0_view,
        inv_work_idx, inv_weight_dev, F_work, d, n_spec,
        is_sep, inv_phase_factors,
        M_tup, N_tup, L_tup
    )
end

# ============================================================================
# Phase 4 — Real-valued (rfft/irfft) planning
# ============================================================================

# ── rfft-aware SoA reconstruction table ──────────────────────────────────────

"""
    _build_recon_soa_rfft(spec_asu, rep_ops, M_sub, M̂, N, dim)

Build SoA reconstruction table for rfft half-spectrum layout.
For frequencies with rot_h[1] >= M̂[1], Hermitian-map to (-rot_h mod M)
and encode conjugation via negative index.

Returns (recon_fiber_idx, recon_weight) with sign-encoded indices.
"""
function _build_recon_soa_rfft(spec_asu::SpecASU, rep_ops::Vector{<:SymOp},
                               M_sub::Vector{Int}, M̂::Vector{Int},
                               N, dim)
    n_spec = length(spec_asu.points)
    n_ops = length(rep_ops)
    total = n_ops * n_spec
    M̂1 = M̂[1]

    recon_fiber_idx = Vector{Int32}(undef, total)
    recon_weight = Vector{ComplexF64}(undef, total)

    rot_h = zeros(Int, dim)

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        for (g_idx, g) in enumerate(rep_ops)
            # Phase: exp(-2πi h·t_g/N)
            phase_val = 0.0
            for d in 1:dim
                phase_val += h_vec[d] * g.t[d] / N[d]
            end
            weight = exp(-im * 2π * phase_val)

            # Rotated frequency: R_g^T h mod M
            for d in 1:dim
                rot_h[d] = mod(sum(g.R[d2, d] * h_vec[d2] for d2 in 1:dim), M_sub[d])
            end

            k = (h_idx - 1) * n_ops + g_idx

            if rot_h[1] < M̂1
                # Direct access in rfft output
                lin_idx = 1 + rot_h[1] + M̂1 * rot_h[2]
                for dd in 3:dim
                    stride = M̂1
                    for dd2 in 2:dd-1
                        stride *= M_sub[dd2]
                    end
                    lin_idx += rot_h[dd] * stride
                end
                recon_fiber_idx[k] = Int32(lin_idx)
                recon_weight[k] = weight
            else
                # Hermitian map: h' = (-rot_h) mod M → guaranteed h'[1] < M̂1
                herm = [mod(-rot_h[d], M_sub[d]) for d in 1:dim]
                lin_idx = 1 + herm[1] + M̂1 * herm[2]
                for dd in 3:dim
                    stride = M̂1
                    for dd2 in 2:dd-1
                        stride *= M_sub[dd2]
                    end
                    lin_idx += herm[dd] * stride
                end
                recon_fiber_idx[k] = Int32(-lin_idx)  # negative = conjugate
                recon_weight[k] = weight
            end
        end
    end

    return recon_fiber_idx, recon_weight
end

# ── plan_real_forward ────────────────────────────────────────────────────────

"""
    plan_real_forward(::Type{T}, spec_asu, ops_shifted; backend=CPU()) → RealForwardPlan

Construct a device-agnostic forward KRFFT plan using rfft for real inputs.
"""
function plan_real_forward(::Type{T}, spec_asu::SpecASU,
                           ops_shifted::Vector{<:SymOp};
                           backend=CPU()) where {T<:AbstractFloat}
    CT = Complex{T}
    N = spec_asu.N
    dim = length(N)

    # 1. Auto L and M
    L_vec = auto_L(ops_shifted)
    M_sub = [N[d] ÷ L_vec[d] for d in 1:dim]
    M_vol = prod(M_sub)
    n_spec = length(spec_asu.points)

    # rfft output dimensions: first dim halved
    M̂_sub = copy(M_sub)
    M̂_sub[1] = M_sub[1] ÷ 2 + 1
    M̂_vol = prod(M̂_sub)

    # 2. Representative ops
    rep_ops = _select_rep_ops(ops_shifted, L_vec, N, dim)
    n_ops = prod(L_vec)

    # 3. SoA reconstruction table (rfft-aware, CPU)
    fiber_idx_cpu, weight_cpu = _build_recon_soa_rfft(
        spec_asu, rep_ops, M_sub, M̂_sub, N, dim)
    weight_cpu_T = CT.(weight_cpu)

    # 4. Buffers (CPU first)
    input_buf_cpu = zeros(T, M_sub...)
    work_buf_cpu  = zeros(CT, M̂_vol)
    out_buf_cpu   = zeros(CT, n_spec)

    # 5. Transfer to device
    input_buf = _to_device(backend, input_buf_cpu)
    work_buf  = _to_device(backend, work_buf_cpu)
    out_buf   = _to_device(backend, out_buf_cpu)
    fiber_idx = _to_device(backend, fiber_idx_cpu)
    weight_dev = _to_device(backend, weight_cpu_T)

    # 6. Views
    M_tup = NTuple{dim, Int}(M_sub)
    M̂_tup = NTuple{dim, Int}(M̂_sub)
    input_view = reshape(input_buf, M_tup)
    work_view  = reshape(work_buf, M̂_tup)

    # 7. rfft plan on device array
    rfft_plan = plan_rfft(input_view)

    # 8. Pmmm detection and phase factors
    is_pmmm = _is_pmmm_pattern(rep_ops, L_vec, N, dim)
    VA = typeof(out_buf)
    if is_pmmm
        phase_cpu = _build_phase_factors(T, M_sub, N, dim)
        phase_factors = VA[_to_device(backend, p) for p in phase_cpu]
    else
        phase_factors = VA[]
    end

    N_tup = NTuple{dim, Int}(N)
    L_tup = NTuple{dim, Int}(L_vec)

    return RealForwardPlan(
        rfft_plan, input_buf, work_buf, out_buf,
        input_view, work_view,
        fiber_idx, weight_dev, n_ops, n_spec,
        is_pmmm, phase_factors,
        M_tup, M̂_tup, N_tup, L_tup
    )
end

# ── rfft-aware inverse reconstruction table ──────────────────────────────────

"""Build inv_recon table for rfft half-spectrum.
Only fills entries for q with q[1] ∈ [0, M̂₁-1]."""
function _build_inv_recon_table_rfft!(inv_spec_idx::Vector{Int32},
                                      inv_weight_cpu::Vector{ComplexF64},
                                      inv_conj_flag::Vector{Bool},
                                      rep_ops::Vector{<:SymOp},
                                      alphas::Vector{NTuple{3,Int}},
                                      h_to_spec::Dict{NTuple{3,Int}, Tuple{Int, ComplexF64, Bool}},
                                      M_sub::Vector{Int}, M̂::Vector{Int},
                                      N, d::Int, dim::Int)
    M̂_vol = prod(M̂)
    B_matrix = zeros(ComplexF64, d, d)
    h_vecs = [zeros(Int, dim) for _ in 1:d]
    ipiv = Vector{Int64}(undef, d)
    rhs = zeros(ComplexF64, d, 1)
    rhs[1] = 1.0
    rhs_work = similar(rhs)
    q_vec = zeros(Int, dim)

    rep_translations = NTuple{3,Int}[(Int(g.t[1]), Int(g.t[2]), Int(g.t[3])) for g in rep_ops]
    N1, N2, N3 = N[1], N[2], N[3]
    M̂_tup = Tuple(M̂)
    ci = CartesianIndices(M̂_tup)
    li = LinearIndices(M̂_tup)

    for q_cart in ci
        for dd in 1:dim
            q_vec[dd] = q_cart[dd] - 1
        end
        q_lin = li[q_cart]

        # Full-grid frequencies in this fiber
        for a in 1:d
            α = alphas[a]
            for dd in 1:dim
                h_vecs[a][dd] = q_vec[dd] + M_sub[dd] * α[dd]
            end
        end

        # Build butterfly matrix
        fill!(B_matrix, zero(ComplexF64))
        for a in 1:d
            h = h_vecs[a]
            for b in 1:d
                t_b = rep_translations[b]
                phase_val = h[1] * t_b[1] / N1 + h[2] * t_b[2] / N2 + h[3] * t_b[3] / N3
                B_matrix[a, b] = cispi(-2 * phase_val)
            end
        end

        # Solve B^T · x = e₁
        copyto!(rhs_work, rhs)
        LAPACK.getrf!(B_matrix, ipiv)
        LAPACK.getrs!('T', B_matrix, ipiv, rhs_work)

        # Map to spectral ASU
        for a in 1:d
            h = h_vecs[a]
            h_key = (mod(h[1], N1), mod(h[2], N2), mod(h[3], N3))
            k = (q_lin - 1) * d + a

            if haskey(h_to_spec, h_key)
                spec_idx, sym_phase, conj_f = h_to_spec[h_key]
                inv_spec_idx[k] = Int32(spec_idx)
                inv_weight_cpu[k] = rhs_work[a] * sym_phase
                inv_conj_flag[k] = conj_f
            else
                inv_spec_idx[k] = Int32(1)
                inv_weight_cpu[k] = zero(ComplexF64)
                inv_conj_flag[k] = false
            end
        end
    end
end

# ── plan_real_backward ───────────────────────────────────────────────────────

"""
    plan_real_backward(::Type{T}, spec_asu, ops_shifted; backend=CPU()) → RealBackwardPlan

Construct a device-agnostic backward KRFFT plan using irfft.
Inverse reconstruction only fills the half-spectrum (M̂₁×M₂×M₃).
"""
function plan_real_backward(::Type{T}, spec_asu::SpecASU,
                            ops_shifted::Vector{<:SymOp};
                            backend=CPU()) where {T<:AbstractFloat}
    CT = Complex{T}
    N = spec_asu.N
    dim = length(N)
    N_vec = collect(N)

    # 1. L and M
    L_vec = auto_L(ops_shifted)
    M_sub = [N[d] ÷ L_vec[d] for d in 1:dim]
    M_vol = prod(M_sub)
    n_spec = length(spec_asu.points)
    d = prod(L_vec)

    # rfft output dimensions
    M̂_sub = copy(M_sub)
    M̂_sub[1] = M_sub[1] ÷ 2 + 1
    M̂_vol = prod(M̂_sub)

    # 2. Representative ops
    rep_ops = _select_rep_ops(ops_shifted, L_vec, N, dim)

    # 3. Spectral reverse lookup
    h_to_spec = _build_spectral_reverse_lookup(spec_asu, ops_shifted, N_vec, dim)

    # 4. Inverse reconstruction table (rfft half-spectrum)
    soa_len = d * M̂_vol
    inv_spec_idx_cpu = Vector{Int32}(undef, soa_len)
    inv_weight_cpu = Vector{ComplexF64}(undef, soa_len)
    inv_conj_flag = Vector{Bool}(undef, soa_len)

    alphas = Vector{NTuple{3,Int}}(undef, d)
    a_idx = 0
    for x0 in Iterators.product([0:L_vec[dd]-1 for dd in 1:dim]...)
        a_idx += 1
        alphas[a_idx] = (x0[1], x0[2], x0[3])
    end
    _build_inv_recon_table_rfft!(inv_spec_idx_cpu, inv_weight_cpu, inv_conj_flag,
                                  rep_ops, alphas, h_to_spec, M_sub, M̂_sub, N, d, dim)

    # 5. SoA work indices (sign-encoded: positive=direct, negative=conjugate)
    inv_work_idx_cpu = Vector{Int32}(undef, soa_len)
    for k in 1:soa_len
        inv_work_idx_cpu[k] = inv_conj_flag[k] ?
            Int32(-inv_spec_idx_cpu[k]) : inv_spec_idx_cpu[k]
    end
    inv_weight_T = CT.(inv_weight_cpu)

    # 6. Buffers
    Y_buf_cpu = zeros(CT, M̂_vol)
    f0_buf_cpu = zeros(T, M_sub...)

    # 7. Transfer to device
    inv_work_idx = _to_device(backend, inv_work_idx_cpu)
    inv_weight_dev = _to_device(backend, inv_weight_T)
    Y_buf = _to_device(backend, Y_buf_cpu)
    f0_buf = _to_device(backend, f0_buf_cpu)

    M_tup = NTuple{dim, Int}(M_sub)
    M̂_tup = NTuple{dim, Int}(M̂_sub)
    Y_view = reshape(Y_buf, M̂_tup)
    f0_view = reshape(f0_buf, M_tup)

    # 8. irfft plan on device array
    irfft_plan = plan_irfft(Y_view, M_sub[1])

    # 9. Pmmm detection
    is_sep = _is_pmmm_pattern(rep_ops, L_vec, N, dim)
    VA = typeof(Y_buf)
    if is_sep
        inv_phase_factors = Vector{VA}(undef, dim)
        for dd in 1:dim
            pf_cpu = [CT(cispi(-2 * q / N[dd])) for q in 0:M_sub[dd]-1]
            inv_phase_factors[dd] = _to_device(backend, pf_cpu)
        end
    else
        inv_phase_factors = VA[]
    end

    N_tup = NTuple{dim, Int}(N)
    L_tup = NTuple{dim, Int}(L_vec)

    return RealBackwardPlan(
        irfft_plan, Y_buf, f0_buf, Y_view, f0_view,
        inv_work_idx, inv_weight_dev, d, n_spec,
        is_sep, inv_phase_factors,
        M_tup, M̂_tup, N_tup, L_tup
    )
end

# ============================================================================
# Phase 3 — Centering fold planning
# ============================================================================

"""
    _alive_offsets(centering) → Vector{NTuple{3,Int}}

Return the parity offsets of alive (non-extinct) frequency classes.
"""
function _alive_offsets(centering::CenteringType)
    if centering == CentI
        return [(0,0,0), (1,1,0), (1,0,1), (0,1,1)]
    elseif centering == CentF
        return [(0,0,0), (1,1,1)]
    elseif centering == CentC
        return [(0,0,0), (1,1,0), (0,0,1), (1,1,1)]
    elseif centering == CentA
        return [(0,0,0), (0,1,1), (1,0,0), (1,1,1)]
    else
        error("Centering fold not applicable for $centering")
    end
end

"""
    plan_centering_fold(::Type{T}, centering, M; backend=CPU()) → CenteringFoldPlan

Construct a device-agnostic centering fold plan for subgrid of dimensions M.
"""
function plan_centering_fold(::Type{T}, centering::CenteringType,
                             M::NTuple{3,Int};
                             backend=CPU()) where {T<:AbstractFloat}
    CT = Complex{T}
    @assert all(iseven, M) "Subgrid dimensions must be even for centering fold"
    H = M .÷ 2

    offsets = _alive_offsets(centering)
    n_ch = length(offsets)

    # Per-channel buffers (CPU path: individual arrays; GPU path: views into batch)
    channel_bufs_cpu = [zeros(CT, H...) for _ in 1:n_ch]
    channel_fft_out_cpu = [zeros(CT, H...) for _ in 1:n_ch]

    # Batched 4D arrays: (H1, H2, H3, n_ch)
    batch_buf_cpu = zeros(CT, H..., n_ch)
    batch_fft_out_cpu = zeros(CT, H..., n_ch)

    # Transfer to device
    batch_buf = _to_device(backend, batch_buf_cpu)
    batch_fft_out = _to_device(backend, batch_fft_out_cpu)

    # Per-channel arrays: on GPU use views into batch arrays; on CPU use independent
    if backend isa CPU
        channel_bufs = [_to_device(backend, channel_bufs_cpu[c]) for c in 1:n_ch]
        channel_fft_out = [_to_device(backend, channel_fft_out_cpu[c]) for c in 1:n_ch]
    else
        channel_bufs = [view(batch_buf, :, :, :, c) for c in 1:n_ch]
        channel_fft_out = [view(batch_fft_out, :, :, :, c) for c in 1:n_ch]
    end

    # Per-channel FFT/IFFT plans (CPU path)
    fft_plans = [plan_fft(channel_bufs[c]) for c in 1:n_ch]
    ifft_plans = [plan_ifft(channel_fft_out[c]) for c in 1:n_ch]

    # Batched FFT/IFFT plans (GPU path: FFT along dims 1,2,3, batch over dim 4)
    batch_fft_plan = plan_fft(batch_buf, (1, 2, 3))
    batch_ifft_plan = plan_ifft(batch_fft_out, (1, 2, 3))

    # Twiddle tables: tw_d[n+1] = cispi(-2 * off_d * n / M_d) for n in 0:H_d-1
    # Compute on CPU, then transfer to device
    twiddle_1d_cpu = Vector{NTuple{3, Vector{CT}}}(undef, n_ch)
    for c in 1:n_ch
        off = offsets[c]
        tw = ntuple(3) do d
            if off[d] == 0
                ones(CT, H[d])
            else
                CT[cispi(T(-2 * off[d] * n / M[d])) for n in 0:H[d]-1]
            end
        end
        twiddle_1d_cpu[c] = tw
    end

    # Transfer twiddle vectors to device
    twiddle_1d = [ntuple(d -> _to_device(backend, twiddle_1d_cpu[c][d]), 3) for c in 1:n_ch]

    # Sign table: signs[(ez*4+ey*2+ex)+1] = (-1)^(off·ε)
    sign_table = Vector{NTuple{8,Int}}(undef, n_ch)
    for c in 1:n_ch
        off = offsets[c]
        signs = ntuple(8) do idx
            ex = (idx - 1) & 1
            ey = ((idx - 1) >> 1) & 1
            ez = ((idx - 1) >> 2) & 1
            1 - 2 * ((off[1]*ex + off[2]*ey + off[3]*ez) & 1)
        end
        sign_table[c] = signs
    end

    # Convert centering enum to Symbol for struct
    cent_sym = centering == CentI ? :I : centering == CentF ? :F :
               centering == CentC ? :C : centering == CentA ? :A : :P

    # Alive mask: parity → channel index (0 = dead, used by fused assemble kernel)
    alive_mask_cpu = zeros(Int32, 8)
    for c in 1:n_ch
        off = offsets[c]
        parity = off[1] + off[2]*2 + off[3]*4
        alive_mask_cpu[parity + 1] = Int32(c)
    end
    alive_mask = _to_device(backend, alive_mask_cpu)

    return CenteringFoldPlan(
        cent_sym, M, H, n_ch, offsets,
        channel_bufs, fft_plans, ifft_plans, channel_fft_out,
        batch_buf, batch_fft_out, batch_fft_plan, batch_ifft_plan,
        twiddle_1d, sign_table,
        alive_mask
    )
end

# ── plan_centered_forward ────────────────────────────────────────────────────

"""
    plan_centered_forward(::Type{T}, spec_asu, ops_shifted; backend=CPU())

Construct a centered forward CFFT plan.
Composes: CenteringFoldPlan + ForwardPlan (inner KRFFT for reconstruction).
"""
function plan_centered_forward(::Type{T}, spec_asu::SpecASU,
                               ops_shifted::Vector{<:SymOp};
                               backend=CPU()) where {T<:AbstractFloat}
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "Centered path only supports 3D"

    # Build inner KRFFT plan (operates on M-grid)
    krfft_plan = plan_forward(T, spec_asu, ops_shifted; backend)
    M = krfft_plan.M

    # Detect centering
    cent = detect_centering_type(ops_shifted, NTuple{dim,Int}(N))
    @assert cent != CentP "Centered path requires non-P centering"

    # Build fold plan
    fold_plan = plan_centering_fold(T, cent, M; backend)

    # Allocate f₀ buffer (real, on device)
    f0_buf_cpu = zeros(T, M...)
    f0_buf = _to_device(backend, f0_buf_cpu)

    # G₀ view = reshape of the inner plan's work_buffer
    G0_view = reshape(krfft_plan.work_buffer, M)

    return CenteredForwardPlan(krfft_plan, fold_plan, f0_buf, G0_view)
end

# ── plan_centered_backward ───────────────────────────────────────────────────

"""
    plan_centered_backward(::Type{T}, spec_asu, ops_shifted; backend=CPU())

Construct a centered backward CFFT plan using CSR orbit-based reconstruction.
"""
function plan_centered_backward(::Type{T}, spec_asu::SpecASU,
                                ops_shifted::Vector{<:SymOp};
                                backend=CPU()) where {T<:AbstractFloat}
    CT = Complex{T}
    N = spec_asu.N
    dim = length(N)
    N_vec = collect(N)
    @assert dim == 3 "Centered path only supports 3D"

    # 1. L, M, d
    L_vec = auto_L(ops_shifted)
    M_sub = [N[d] ÷ L_vec[d] for d in 1:dim]
    M_vol = prod(M_sub)
    n_spec = length(spec_asu.points)
    d = prod(L_vec)

    # 2. Representative ops
    rep_ops = _select_rep_ops(ops_shifted, L_vec, N, dim)

    # 3. Spectral reverse lookup
    h_to_spec = _build_spectral_reverse_lookup(spec_asu, ops_shifted, N_vec, dim)

    # 4. Build M2-style inv_recon (all positions)
    alphas = Vector{NTuple{3,Int}}(undef, d)
    a_idx = 0
    for x0 in Iterators.product([0:L_vec[dd]-1 for dd in 1:dim]...)
        a_idx += 1
        alphas[a_idx] = (x0[1], x0[2], x0[3])
    end

    # Full inv_recon table (spec_idx, weight, conj_flag per fiber entry)
    full_spec_idx = Vector{Int32}(undef, d * M_vol)
    full_weight = Vector{ComplexF64}(undef, d * M_vol)
    full_conj_flag = Vector{Bool}(undef, d * M_vol)
    _build_inv_recon_table!(full_spec_idx, full_weight, full_conj_flag,
                            rep_ops, alphas, h_to_spec, M_sub, N, d, dim)

    # 5. Build spatial orbits under G_rem (even-translation ops)
    M_tup = NTuple{dim, Int}(M_sub)
    rem_ops = Tuple{Matrix{Int}, Vector{Int}}[]
    for op in ops_shifted
        t_int = round.(Int, op.t)
        if all(mod.(t_int, 2) .== 0)
            push!(rem_ops, (round.(Int, op.R), t_int .÷ 2))
        end
    end

    orbit_id = zeros(Int32, M_vol)
    orbit_phase = zeros(ComplexF64, M_vol)
    orbits_rep = Int32[]

    ci = CartesianIndices(M_tup)
    li = LinearIndices(M_tup)

    for m in 1:M_vol
        orbit_id[m] != 0 && continue
        push!(orbits_rep, Int32(m))
        oid = Int32(length(orbits_rep))
        orbit_id[m] = oid
        orbit_phase[m] = complex(1.0)

        # BFS orbit expansion with phase tracking
        queue = [m]
        while !isempty(queue)
            cur = popfirst!(queue)
            cur0 = cur - 1
            qv = (mod(cur0, M_sub[1]),
                  mod(div(cur0, M_sub[1]), M_sub[2]),
                  div(cur0, M_sub[1] * M_sub[2]))
            cur_phase = orbit_phase[cur]

            for (R, s) in rem_ops
                # Apply R^T to frequency q
                rq = ntuple(d1 -> mod(sum(R[d2, d1] * qv[d2] for d2 in 1:dim),
                                      M_sub[d1]), Val(3))
                rq_lin = 1 + rq[1] + M_sub[1] * rq[2] + M_sub[1] * M_sub[2] * rq[3]

                if orbit_id[rq_lin] == 0
                    ph = cispi(2.0 * sum(qv[dd] * s[dd] / M_sub[dd] for dd in 1:dim))
                    orbit_id[rq_lin] = oid
                    orbit_phase[rq_lin] = ph * cur_phase
                    push!(queue, rq_lin)
                end
            end
        end
    end

    n_orbits = length(orbits_rep)

    # 6. Build CSR compact table for orbit reps only
    offsets_csr = Vector{Int32}(undef, n_orbits + 1)
    offsets_csr[1] = Int32(1)
    for i in 1:n_orbits
        q = orbits_rep[i]
        cnt = Int32(0)
        for a in 1:d
            k = (q - 1) * d + a
            abs(full_weight[k]) > 1e-15 && (cnt += 1)
        end
        offsets_csr[i + 1] = offsets_csr[i] + cnt
    end

    nnz = Int(offsets_csr[n_orbits + 1] - 1)
    csr_spec_idx = Vector{Int32}(undef, nnz)
    csr_weight = Vector{ComplexF64}(undef, nnz)

    for i in 1:n_orbits
        q = orbits_rep[i]
        j = Int(offsets_csr[i])
        for a in 1:d
            k = (q - 1) * d + a
            abs(full_weight[k]) > 1e-15 || continue
            # Negative spec_idx signals conjugation at runtime
            csr_spec_idx[j] = full_conj_flag[k] ? -full_spec_idx[k] : full_spec_idx[k]
            csr_weight[j] = full_weight[k]
            j += 1
        end
    end

    # 7. Centering fold plan
    cent = detect_centering_type(ops_shifted, NTuple{dim,Int}(N))
    fold_plan = plan_centering_fold(T, cent, M_tup; backend)

    # 8. Buffers
    G0_reps_cpu = zeros(CT, n_orbits)
    f0_buf_cpu = zeros(T, M_tup...)
    # G0 view shares the work_buffer concept — allocate a dedicated M³ complex buffer
    G0_buf_cpu = zeros(CT, M_vol)

    # 9. Transfer to device
    inv_offsets = _to_device(backend, offsets_csr)
    inv_spec_idx = _to_device(backend, csr_spec_idx)
    inv_weight_dev = _to_device(backend, CT.(csr_weight))
    orbit_rep_dev = _to_device(backend, orbits_rep)
    orbit_oid_dev = _to_device(backend, orbit_id)
    orbit_phase_dev = _to_device(backend, CT.(orbit_phase))
    G0_reps_dev = _to_device(backend, G0_reps_cpu)
    f0_buf = _to_device(backend, f0_buf_cpu)
    G0_buf = _to_device(backend, G0_buf_cpu)
    G0_view = reshape(G0_buf, M_tup)

    N_tup = NTuple{dim, Int}(N)

    return CenteredBackwardPlan(
        inv_offsets, inv_spec_idx, inv_weight_dev, n_orbits,
        orbit_rep_dev, orbit_oid_dev, orbit_phase_dev, G0_reps_dev,
        fold_plan, f0_buf, G0_view,
        M_tup, n_spec
    )
end
