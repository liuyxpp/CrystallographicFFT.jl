# ============================================================================
# G0 ASU Backward Transform — spectral ASU → real-space
# ============================================================================

"""
    ExpandEntry

A precomputed entry for spectral ASU expansion: maps a spectral point to
a full-grid frequency position.
"""
struct ExpandEntry
    full_lin::Int32     # 1-based linear index into N³ full spectrum
    phase::ComplexF64   # exp(-2πi h·t/N) phase factor
end

"""
    G0ASUBackwardPlan

Plan for inverse G0 ASU KRFFT: spectral ASU → real-space array.

Pipeline (via full-grid IFFT):
1. Expand spectral ASU → full N³ frequency spectrum using symmetry
2. IFFT on full N³ grid
3. Take real part

For a symmetrized field, the symmetry operations already map each spectral
ASU point to ALL its symmetry-equivalent frequencies (including Hermitian
partners), so no separate Hermitian conjugate step is needed.
"""
struct G0ASUBackwardPlan
    ifft_plan::Any                          # FFTW IFFT plan for N³

    # Full-grid buffer for IFFT
    F_full::Array{ComplexF64,3}             # N³ full frequency spectrum

    # Spectral expansion table (CSR-like)
    expand_entries::Vector{ExpandEntry}
    expand_row_ptrs::Vector{Int}            # n_spec + 1 entries

    # Grid dimensions
    grid_N::Vector{Int}
    n_spec::Int
end

"""
    plan_krfft_g0asu_backward(spec_asu, ops_shifted)

Create a backward plan for G0 ASU KRFFT (spectral ASU → real-space).

For each spectral ASU point h, all symmetry operations g ∈ G produce
frequencies at q = R_g^T h mod N with phase exp(-2πi h·t_g/N).

The spectral ASU is chosen so that this expansion covers all non-zero
frequency positions exactly once (no collisions, no gaps).
"""
function plan_krfft_g0asu_backward(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "G0 ASU backward currently supports 3D only"

    n_spec = length(spec_asu.points)

    # Build spectral expansion table
    all_expand = ExpandEntry[]
    row_ptrs = Vector{Int}(undef, n_spec + 1)

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        row_ptrs[h_idx] = length(all_expand) + 1

        # Track positions already written for this spectral point
        # (different ops may map to the same q — use first only)
        positions_seen = Set{Int}()

        for op in ops_shifted
            # Rotated frequency: q = R^T h mod N (in full grid)
            q = [mod(sum(round(Int, op.R[d2, d1]) * h_vec[d2] for d2 in 1:dim), N[d1]) for d1 in 1:dim]
            lin = 1 + q[1] + N[1] * q[2] + N[1] * N[2] * q[3]

            if lin ∉ positions_seen
                push!(positions_seen, lin)

                # Phase: exp(+2πi h·t/N) — conjugate of forward phase
                # Forward uses exp(-2πi h·t/N), backward symmetry relation uses +
                phase_val = sum(h_vec[d] * op.t[d] / N[d] for d in 1:dim)
                phase = cispi(+2 * phase_val)

                push!(all_expand, ExpandEntry(Int32(lin), phase))
            end
        end
    end
    row_ptrs[n_spec + 1] = length(all_expand) + 1

    # IFFT plan on full grid
    N_tuple = Tuple(N)
    F_full = zeros(ComplexF64, N_tuple)
    ifft_plan = plan_ifft(F_full)

    @info "G0 ASU backward plan: n_spec=$n_spec, expand_nnz=$(length(all_expand)), N=$(N_tuple)"

    return G0ASUBackwardPlan(
        ifft_plan,
        F_full,
        all_expand, row_ptrs,
        collect(N), n_spec
    )
end

"""
    execute_g0asu_ikrfft!(plan, spec_asu, F_spec, u_out)

Execute backward G0 ASU KRFFT: spectral ASU → real-space.

Pipeline:
1. Expand spectral ASU → full N³ frequency spectrum
2. IFFT on full N³ grid
3. Copy real part to output
"""
function execute_g0asu_ikrfft!(plan::G0ASUBackwardPlan,
                               spec_asu::SpectralIndexing,
                               F_spec::AbstractVector{ComplexF64},
                               u_out::AbstractArray{<:Number,3})
    F_full = plan.F_full
    fill!(F_full, zero(ComplexF64))

    entries = plan.expand_entries
    ptrs = plan.expand_row_ptrs
    n_spec = plan.n_spec

    # 1. Expand spectral ASU → full N³ spectrum
    @inbounds for h in 1:n_spec
        fh = F_spec[h]
        for idx in ptrs[h]:(ptrs[h+1]-1)
            e = entries[idx]
            F_full[e.full_lin] = e.phase * fh
        end
    end

    # 2. IFFT
    u_complex = plan.ifft_plan * F_full

    # 3. Copy real part to output
    @inbounds for i in eachindex(u_out, u_complex)
        u_out[i] = real(u_complex[i])
    end

    return u_out
end
