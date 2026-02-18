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

using .SymmetryOps: SymOp, CenteringType, CentP, detect_centering_type
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

    # 5. FFT plan (on CPU array with target precision)
    M_tup = NTuple{dim, Int}(M_sub)
    dummy = zeros(CT, M_tup)
    fft_plan = plan_fft(dummy)

    # 6. Pmmm detection and phase factors
    is_pmmm = _is_pmmm_pattern(rep_ops, L_vec, N, dim)
    if is_pmmm
        phase_factors = _build_phase_factors(T, M_sub, N, dim)
    else
        phase_factors = Vector{CT}[]
    end

    # 7. Transfer to device
    input_buf = _to_device(backend, input_buf_cpu)
    work_buf  = _to_device(backend, work_buf_cpu)
    out_buf   = _to_device(backend, out_buf_cpu)
    fiber_idx = _to_device(backend, fiber_idx_cpu)
    weight_dev = _to_device(backend, weight_cpu_T)

    # 8. Reshape views
    input_view = reshape(input_buf, M_tup)
    work_view  = reshape(work_buf, M_tup)

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

    # 7. IFFT plan
    M_tup = NTuple{dim, Int}(M_sub)
    dummy = zeros(CT, M_tup)
    ifft_plan = plan_ifft(dummy)

    # 8. Pmmm detection
    is_sep = _is_pmmm_pattern(rep_ops, L_vec, N, dim)
    inv_phase_factors = Vector{CT}[]
    if is_sep
        inv_phase_factors = Vector{Vector{CT}}(undef, dim)
        for dd in 1:dim
            inv_phase_factors[dd] = [CT(cispi(-2 * q / N[dd])) for q in 0:M_sub[dd]-1]
        end
    end

    # 9. Transfer to device
    inv_work_idx = _to_device(backend, inv_work_idx_cpu)
    inv_weight_dev = _to_device(backend, inv_weight_T)
    F_work = _to_device(backend, F_work_cpu)
    Y_buf = _to_device(backend, Y_buf_cpu)
    f0_buf = _to_device(backend, f0_buf_cpu)

    Y_view = reshape(Y_buf, M_tup)
    f0_view = reshape(f0_buf, M_tup)

    N_tup = NTuple{dim, Int}(N)
    L_tup = NTuple{dim, Int}(L_vec)

    return BackwardPlan(
        ifft_plan, Y_buf, f0_buf, Y_view, f0_view,
        inv_work_idx, inv_weight_dev, F_work, d, n_spec,
        is_sep, inv_phase_factors,
        M_tup, N_tup, L_tup
    )
end
