module KRFFT

using KernelAbstractions
using AbstractFFTs
using FFTW
using LinearAlgebra
using SparseArrays
using ..ASU: CrystallographicASU, ASUBlock
using ..SpectralIndexing: SpectralIndexing, get_k_vector
using ..SymmetryOps: SymOp

export GeneralCFFTPlan, map_fft!, map_ifft!, build_recombination_map, plan_krfft
export fast_reconstruct!, pack_stride!, execute_krfft!, ReconEntry, fft_reconstruct!
export auto_L

"""
    ActiveBlock

 Represents a specific (Block, ModulationShift) pair that needs to be computed.
"""
struct ActiveBlock
    block_idx::Int
    r_shift::Vector{Int} # Remainder/Shift indices
    buffer_offset::Int
end

"""
    ReconEntry

Precomputed reconstruction info for one (spectral_point, symmetry_op) pair.
Used by the dense reconstruction loop to replace sparse matmul.
"""
struct ReconEntry
    buffer_idx::Int        # Index into work_buffer (1-based)
    weight::ComplexF64     # Phase factor
end

"""
    GeneralCFFTPlan

A plan for General KRFFT using Modulated Block FFTs (Cooley-Tukey Decomposition).
Decomposes Global frequency q = m * L + r.
"""
struct GeneralCFFTPlan{T, N, P, M, B} <: AbstractFFTs.Plan{T}
    # 1. Sub-plans for small FFTs (shared by size)
    sub_plans::Vector{P}
    
    # 2. Recombination Map (sparse, for backward compat)
    recombination_map::M
    
    # 3. Work Buffer (Stores FFT coeffs of all ActiveBlocks)
    work_buffer::B
    
    # 4. Dense reconstruction table (fast path)
    recon_table::Matrix{ReconEntry}
    output_buffer::Vector{ComplexF64}
    input_buffer::Vector{ComplexF64}
    
    # 5. Precomputed 1D phase factors for separable butterfly
    # phase_factors[d][h+1] = exp(2πi h / N_d) for h = 0..M_d-1
    phase_factors::Vector{Vector{ComplexF64}}
    
    # Metadata
    active_blocks::Vector{ActiveBlock}
    block_dims::Vector{Tuple}
    L_factors::Vector{Vector{Int}}
    n_ops::Int
    grid_N::Vector{Int}
    subgrid_dims::Vector{Int}
end

# Constructor
function GeneralCFFTPlan(sub_plans::Vector{P}, recomb::M, buffer::B,
        recon_table::Matrix{ReconEntry}, output_buffer::Vector{ComplexF64}, input_buffer::Vector{ComplexF64},
        phase_factors::Vector{Vector{ComplexF64}},
        active::Vector{ActiveBlock}, b_dims, Ls, n_ops,
        grid_N::Vector{Int}, subgrid_dims::Vector{Int}) where {P, M, B}
    T = eltype(buffer)
    N = 1
    return GeneralCFFTPlan{T, N, P, M, B}(sub_plans, recomb, buffer,
        recon_table, output_buffer, input_buffer, phase_factors,
        active, b_dims, Ls, n_ops, grid_N, subgrid_dims)
end

function plan_krfft(real_asu::CrystallographicASU, spec_asu::SpectralIndexing, direct_ops::Vector{SymOp})
    # 1. Analyze Blocks
    all_blocks = Vector{ASUBlock}()
    for d in sort(collect(keys(real_asu.dim_blocks)))
        append!(all_blocks, real_asu.dim_blocks[d])
    end
    
    N = spec_asu.N
    dim = length(N)
    n_ops = length(direct_ops)
    
    L_factors = Vector{Vector{Int}}()
    block_dims_list = Vector{Tuple}()
    
    for b in all_blocks
        dims = size(b.data)
        push!(block_dims_list, dims)
        L = [N[d] ÷ dims[d] for d in 1:dim]
        if any(L .* dims .!= N)
            error("Block size must divide Global size for Cooley-Tukey decomposition")
        end
        push!(L_factors, L)
    end
    
    # 2. Build Recombination Map & Identify Active Blocks
    M_recomb, active_blocks, buffer_size = build_recombination_map(spec_asu, real_asu, direct_ops, all_blocks, L_factors)
    
    # 3. Plan FFTs (out-of-place for better FFTW performance)
    block_plans = []
    for b in all_blocks
        dummy_in = zeros(ComplexF64, size(b.data))
        push!(block_plans, plan_fft(dummy_in))
    end
    
    # 4. Alloc Buffers
    work_buffer = zeros(ComplexF64, buffer_size)
    
    # 5. Build dense reconstruction table (fast path)
    M_sub = collect(size(all_blocks[1].data))  # Subgrid dims
    n_spec = length(spec_asu.points)
    recon_table = Matrix{ReconEntry}(undef, n_ops, n_spec)
    
    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        
        for (g_idx, g) in enumerate(direct_ops)
            # Phase: exp(-2πi h·t_g/N)  (includes shift implicitly via t_g)
            phase_val = 0.0
            for d in 1:dim
                phase_val += h_vec[d] * g.t[d] / N[d]
            end
            weight = exp(-im * 2π * phase_val)
            
            # Rotated frequency: R_g^T h mod M
            rot_h = [mod(sum(g.R[d2, d1] * h_vec[d2] for d2 in 1:dim), M_sub[d1]) for d1 in 1:dim]
            
            # Linear index into work_buffer (1-based)
            lin_idx = 1
            stride = 1
            for d in 1:dim
                lin_idx += rot_h[d] * stride
                stride *= M_sub[d]
            end
            
            recon_table[g_idx, h_idx] = ReconEntry(lin_idx, weight)
        end
    end
    
    output_buffer = zeros(ComplexF64, n_spec)
    input_buffer = zeros(ComplexF64, Tuple(M_sub)...)
    
    # Precompute 1D phase factors: φ_d[h+1] = exp(2πi h / N_d) for h = 0..M_d-1
    # For Pmmm shifted ops: t_g[d] = -1 when R[d,d]=-1, t_g[d] = 0 when R[d,d]=+1
    # So phase = exp(-2πi h_d * (-1) / N_d) = exp(2πi h_d / N_d)
    phase_factors = Vector{Vector{ComplexF64}}(undef, dim)
    for d in 1:dim
        phase_factors[d] = [cispi(2 * h / N[d]) for h in 0:M_sub[d]-1]
    end
    
    P_type = eltype(block_plans)
    
    return GeneralCFFTPlan(convert(Vector{P_type}, block_plans), M_recomb, work_buffer,
        recon_table, output_buffer, vec(input_buffer), phase_factors,
        active_blocks, block_dims_list, L_factors, n_ops,
        collect(N), M_sub)
end

"""
    auto_L(ops_shifted::Vector{SymOp}) -> Vector{Int}

Determine optimal Phase 1 Cooley-Tukey factor L from shifted symmetry operations.

For each dimension d, L_d = 2 if any shifted operation has t_d odd (i.e., the 
operation flips the subgrid parity in that dimension). Otherwise L_d = 1.

When prod(L) exceeds the number of reachable subgrids (i.e., |G| is too small 
to cover all 2^D subgrids), L is automatically reduced to ensure n_active = 1
(only one independent FFT needed).

# Examples
```julia
auto_L(ops_pmmm)   # → [2, 2, 2]  (8x)
auto_L(ops_pmm2)   # → [2, 2, 1]  (4x)
auto_L(ops_pm)     # → [1, 2, 1]  (2x)
```
"""
function auto_L(ops_shifted::Vector{SymOp})
    dim = length(ops_shifted[1].t)
    
    # Step 1: Determine L per dimension from translation parity
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
    if n_subgrids <= 1
        return L_max
    end
    
    # Step 2: Count reachable subgrids
    reachable = Set{Vector{Int}}()
    for op in ops_shifted
        t = round.(Int, op.t)
        parity = [mod(t[d], L_max[d]) for d in 1:dim]
        push!(reachable, parity)
    end
    
    if length(reachable) == n_subgrids
        return L_max  # All subgrids reachable → optimal
    end
    
    # Step 3: Reduce L to maximize speedup with n_active = 1
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

"""
    plan_krfft(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp}) -> GeneralCFFTPlan

Auto-L variant: computes optimal L from ops, constructs plan without pre-packed ASU.

This is the preferred entry point for SCFT workflows. The subgrid dimensions M are
determined as N .÷ L, and all buffers are allocated internally.

# Usage
```julia
ops = get_ops(47, 3, N)
_, ops_shifted = find_optimal_shift(ops, N)
spec = calc_spectral_asu(ops_shifted, 3, N)
plan = plan_krfft(spec, ops_shifted)

# SCFT fast path:
plan.input_buffer .= vec(subgrid_data)
F = fft_reconstruct!(plan)
```
"""
function plan_krfft(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    
    # 1. Auto-determine L
    L = auto_L(ops_shifted)
    M_sub = [N[d] ÷ L[d] for d in 1:dim]
    
    if any(L .* M_sub .!= collect(N))
        error("Grid size N=$N not divisible by auto L=$L. Use N that is a multiple of L in each dimension.")
    end
    
    # 2. Select one representative operation per subgrid
    #    Group ops by subgrid mapping: x₀ = t_g mod L
    #    F(h) = Σ_{x₀} exp(-2πi h·t_{g(x₀)}/N) · Y₀(R_{g(x₀)}^T h mod M)
    #    Prefer diagonal R with simple translations t_d ∈ {0,-1} so more groups
    #    can benefit from the separable Pmmm fast path.
    subgrid_reps = Dict{Vector{Int}, SymOp}()
    subgrid_quality = Dict{Vector{Int}, Int}()  # higher = better
    
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], L[d]) for d in 1:dim]
        
        # Score: 2 = diagonal R + simple t, 1 = diagonal R only, 0 = non-diagonal
        is_diag = all(op.R[i,j] == 0 for i in 1:dim for j in 1:dim if i != j)
        simple_t = all(mod(t[d], N[d]) ∈ (0, N[d]-1) for d in 1:dim)
        quality = is_diag ? (simple_t ? 2 : 1) : 0
        
        if !haskey(subgrid_reps, x0) || quality > subgrid_quality[x0]
            subgrid_reps[x0] = op
            subgrid_quality[x0] = quality
        end
    end
    
    # Enumerate subgrids in canonical order
    n_subs = prod(L)
    rep_ops = Vector{SymOp}(undef, n_subs)
    sub_idx = 0
    for x0 in Iterators.product([0:L[d]-1 for d in 1:dim]...)
        sub_idx += 1
        x0_vec = collect(x0)
        if haskey(subgrid_reps, x0_vec)
            rep_ops[sub_idx] = subgrid_reps[x0_vec]
        else
            error("Subgrid x₀=$x0_vec not reachable from subgrid 0. auto_L should have prevented this.")
        end
    end
    
    n_ops_effective = n_subs  # Always prod(L), not |G|
    
    # 3. Plan out-of-place FFT on subgrid
    dummy_in = zeros(ComplexF64, Tuple(M_sub))
    fft_plan = plan_fft(dummy_in)
    block_plans = [fft_plan]
    
    # 4. Allocate buffers
    buffer_size = prod(M_sub)
    work_buffer = zeros(ComplexF64, buffer_size)
    
    # 5. Build dense reconstruction table using representative ops
    n_spec = length(spec_asu.points)
    recon_table = Matrix{ReconEntry}(undef, n_ops_effective, n_spec)
    
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
            rot_h = [mod(sum(g.R[d2, d1] * h_vec[d2] for d2 in 1:dim), M_sub[d1]) for d1 in 1:dim]
            
            # Linear index into work_buffer (1-based, column-major)
            lin_idx = 1
            stride = 1
            for d in 1:dim
                lin_idx += rot_h[d] * stride
                stride *= M_sub[d]
            end
            
            recon_table[g_idx, h_idx] = ReconEntry(lin_idx, weight)
        end
    end
    
    output_buffer = zeros(ComplexF64, n_spec)
    input_buffer = zeros(ComplexF64, Tuple(M_sub)...)
    
    # 6. Precompute 1D phase factors — only for Pmmm-like groups
    #    Requires: all diagonal R AND t_d ∈ {0, -1} (simple mirror pattern)
    all_diagonal = all(op -> all(op.R[i,j] == 0 for i in 1:dim for j in 1:dim if i != j), rep_ops)
    simple_translations = all(op -> all(round(Int, op.t[d]) ∈ (0, -1) || mod(round(Int, op.t[d]), N[d]) ∈ (0, N[d]-1) for d in 1:dim), rep_ops)
    
    if all_diagonal && simple_translations
        phase_factors = Vector{Vector{ComplexF64}}(undef, dim)
        for d in 1:dim
            phase_factors[d] = [cispi(2 * h / N[d]) for h in 0:M_sub[d]-1]
        end
    else
        # Non-Pmmm pattern: use general recon_table path
        phase_factors = Vector{ComplexF64}[]
    end
    
    # 7. Create active block and L_factors for struct
    active_blocks = [ActiveBlock(1, zeros(Int, dim), 1)]
    block_dims_list = [Tuple(M_sub)]
    L_factors = [L]
    
    P_type = eltype(block_plans)
    
    # Dummy sparse recomb (unused in fast path, kept for interface compat)
    M_recomb = sparse(Int[], Int[], ComplexF64[], n_spec, buffer_size)
    
    return GeneralCFFTPlan(convert(Vector{P_type}, block_plans), M_recomb, work_buffer,
        recon_table, output_buffer, vec(input_buffer), phase_factors,
        active_blocks, block_dims_list, L_factors, n_ops_effective,
        collect(N), M_sub)
end

function map_fft!(plan::GeneralCFFTPlan, asu::CrystallographicASU)
    # 1. Flatten ASU blocks to linear access if needed? No, access by index.
    # Better: Re-linearize `all_blocks` references
    all_blocks = Vector{ASUBlock}()
    for d in sort(collect(keys(asu.dim_blocks)))
        append!(all_blocks, asu.dim_blocks[d])
    end
    
    # 2. Execute Modulated FFTs
    for ab in plan.active_blocks
        # Source Block
        src = all_blocks[ab.block_idx]
        dims = size(src.data)
        len = length(src.data)
        
        # Target Slice in Buffer
        range = ab.buffer_offset:(ab.buffer_offset + len - 1)
        sub_buf = view(plan.work_buffer, range)
        
        # Reshape for Cartesian Access
        sub_grid = reshape(sub_buf, dims)
        
        # Copy & Modulate
        # Modulation: exp(-2pi i * (r ./ N) . n)
        # r is vector of remainders.
        # n is 0-based index in block.
        # factor per dim: exp(-2pi i * r[d]/N[d] * n[d])
        #               = exp(-2pi i * r[d]/(L[d]*Nb[d]) * n[d])
        #               = exp(-2pi i * (r[d]/L[d]) * (n[d]/Nb[d]))
        
        r_vec = ab.r_shift
        L_vec = plan.L_factors[ab.block_idx]
        
        # Fast Path: If r = [0,0,0], skip modulation (Mode B typically has r=0).
        if all(r_vec .== 0)
            # Direct copy (no modulation)
            copyto!(sub_grid, src.data)
        else
            # Slow Path: Modulate (Mode A with r != 0)
            for i in CartesianIndices(src.data)
                val = src.data[i]
                
                # Modulation
                n_idx = Tuple(i) .- 1
                phase = 0.0
                for d in 1:length(dims)
                    phase += r_vec[d] * n_idx[d] / (L_vec[d] * dims[d])
                end
                
                sub_grid[i] = val * exp(-im * 2π * phase)
            end
        end
        
        # Execute FFT in-place (on sub_buf)
        # For FFTW in-place plans, p * sub_grid modifies sub_grid in-place AND returns result
        p = plan.sub_plans[ab.block_idx]
        p * sub_grid  # Result stored in sub_grid (which is view into work_buffer)
    end
    
    # Buffer now contains all needed spectral fragments.
end

"""
    fast_reconstruct!(plan)

Reconstruct spectral ASU from subgrid FFT in work_buffer.
For diagonal-R groups (Pmmm), uses on-the-fly index computation
with precomputed 1D phase tables instead of general ReconEntry table.
Falls back to table-based approach for non-diagonal groups.
"""
function fast_reconstruct!(plan::GeneralCFFTPlan)
    M = plan.subgrid_dims
    dim = length(M)
    
    if dim == 3 && plan.n_ops == 8 && length(plan.phase_factors) == 3 &&
       length(plan.output_buffer) == prod(plan.subgrid_dims)
        return _fast_reconstruct_pmmm!(plan)
    end
    
    # General fallback
    return _fast_reconstruct_general!(plan)
end

"""Specialized Pmmm reconstruction with on-the-fly mirror indices."""
function _fast_reconstruct_pmmm!(plan::GeneralCFFTPlan)
    M1, M2, M3 = plan.subgrid_dims[1], plan.subgrid_dims[2], plan.subgrid_dims[3]
    Y = reshape(plan.work_buffer, M1, M2, M3)
    out = plan.output_buffer
    φ1 = plan.phase_factors[1]
    φ2 = plan.phase_factors[2]
    φ3 = plan.phase_factors[3]
    
    # Spectral ASU enumerates in row-major order: h₁ outermost, h₃ innermost
    h_idx = 0
    @inbounds for i in 1:M1
        i_flip = mod(M1 - (i-1), M1) + 1
        φ1i = φ1[i]  # phase when s₁=-1
        
        for j in 1:M2
            j_flip = mod(M2 - (j-1), M2) + 1
            φ2j = φ2[j]
            
            # Precompute dim-1/dim-2 phase combinations
            w_pp = complex(1.0)   # s1=+1, s2=+1
            w_mp = φ2j            # s1=+1, s2=-1
            w_fp = φ1i            # s1=-1, s2=+1
            w_fm = φ1i * φ2j      # s1=-1, s2=-1
            
            for k in 1:M3
                h_idx += 1
                k_flip = mod(M3 - (k-1), M3) + 1
                φ3k = φ3[k]  # phase when s₃=-1
                
                # 8 Y values at mirrored positions
                y_ppp = Y[i, j, k]
                y_mpp = Y[i_flip, j, k]
                y_pmp = Y[i, j_flip, k]
                y_ppm = Y[i, j, k_flip]
                y_mmp = Y[i_flip, j_flip, k]
                y_mpm = Y[i_flip, j, k_flip]
                y_pmm = Y[i, j_flip, k_flip]
                y_mmm = Y[i_flip, j_flip, k_flip]
                
                # s₃=+1 terms: no φ₃ factor
                sum_p = w_pp * y_ppp + w_fp * y_mpp +
                        w_mp * y_pmp + w_fm * y_mmp
                # s₃=-1 terms: multiply by φ₃
                sum_m = w_pp * y_ppm + w_fp * y_mpm +
                        w_mp * y_pmm + w_fm * y_mmm
                
                out[h_idx] = sum_p + φ3k * sum_m
            end
        end
    end
    return out
end

"""General table-based fallback for non-diagonal groups."""
function _fast_reconstruct_general!(plan::GeneralCFFTPlan)
    n_ops = plan.n_ops
    n_spec = length(plan.output_buffer)
    buf = plan.work_buffer
    table = plan.recon_table
    out = plan.output_buffer
    
    @inbounds for h_idx in 1:n_spec
        val = zero(ComplexF64)
        for g_idx in 1:n_ops
            entry = table[g_idx, h_idx]
            val += entry.weight * buf[entry.buffer_idx]
        end
        out[h_idx] = val
    end
    return out
end

"""
    fft_reconstruct!(plan)

Combined FFT + reconstruct for SCFT fast path.
Assumes subgrid data is already stored as complex values in plan.work_buffer.
Returns plan.output_buffer containing spectral ASU values.
"""
function fft_reconstruct!(plan::GeneralCFFTPlan)
    # 1. Out-of-place FFT: input_buffer → work_buffer
    sub_in = reshape(plan.input_buffer, Tuple(plan.subgrid_dims))
    sub_out = reshape(plan.work_buffer, Tuple(plan.subgrid_dims))
    mul!(sub_out, plan.sub_plans[1], sub_in)
    
    # 2. Reconstruct spectral ASU
    fast_reconstruct!(plan)
    
    return plan.output_buffer
end


"""
    pack_stride!(plan, u)

Fast stride-2 subgrid extraction from real array u into plan's work_buffer.
Directly copies u[1:2:end, 1:2:end, 1:2:end] into the complex work buffer.
"""
function pack_stride!(plan::GeneralCFFTPlan, u::AbstractArray{<:Real})
    N = plan.grid_N
    M = plan.subgrid_dims
    dim = length(N)
    buf = plan.work_buffer
    
    if dim == 3
        # Fast 3D path
        idx = 1
        @inbounds for k in 1:M[3]
            kk = 2*(k-1) + 1  # 1-based index into u
            for j in 1:M[2]
                jj = 2*(j-1) + 1
                for i in 1:M[1]
                    ii = 2*(i-1) + 1
                    buf[idx] = complex(u[ii, jj, kk])
                    idx += 1
                end
            end
        end
    else
        # Generic fallback
        idx = 1
        for ci in CartesianIndices(Tuple(M))
            ui = CartesianIndex(Tuple(2 .* (Tuple(ci) .- 1) .+ 1))
            @inbounds buf[idx] = complex(u[ui])
            idx += 1
        end
    end
end

"""
    execute_krfft!(plan, u)

Full KRFFT pipeline: pack → FFT → reconstruct.
Returns the output_buffer containing spectral ASU values.
"""
function execute_krfft!(plan::GeneralCFFTPlan, u::AbstractArray{<:Real})
    # 1. Pack: stride-2 copy into work_buffer
    pack_stride!(plan, u)
    
    # 2. FFT in-place on work_buffer (reshaped to subgrid dims)
    sub_grid = reshape(plan.work_buffer, Tuple(plan.subgrid_dims))
    p = plan.sub_plans[1]
    p * sub_grid  # in-place FFT
    
    # 3. Reconstruct: dense table loop
    fast_reconstruct!(plan)
    
    return plan.output_buffer
end


function map_ifft!(plan::GeneralCFFTPlan, asu::CrystallographicASU)
    # Inverse:
    # 1. Input: u_spec (Caller uses M') -> plan.work_buffer.
    # Buffer contains summed contributions for Modulated FFTs.
    
    all_blocks = Vector{ASUBlock}()
    for d in sort(collect(keys(asu.dim_blocks)))
        append!(all_blocks, asu.dim_blocks[d])
    end
    
    # Zero out real blocks (accumulation)
    for b in all_blocks
        fill!(b.data, 0.0)
    end
    
    # 2. Execute Modulated IFFTs & Demodulate & Accumulate
    for ab in plan.active_blocks
        src = all_blocks[ab.block_idx]
        dims = size(src.data)
        len = length(src.data)
        
        range = ab.buffer_offset:(ab.buffer_offset + len - 1)
        sub_buf = view(plan.work_buffer, range)
        sub_grid = reshape(sub_buf, dims)
        
        # IFFT
        p = plan.sub_plans[ab.block_idx]
        inv(p) * sub_grid
        
        # Demodulate & Accumulate
        r_vec = ab.r_shift
        L_vec = plan.L_factors[ab.block_idx]
        
        for i in CartesianIndices(src.data)
            # Inverse Modulation: exp(+2pi i ...)
            n_idx = Tuple(i) .- 1
            phase = 0.0
            for d in 1:length(dims)
                 phase += r_vec[d] * n_idx[d] / (L_vec[d] * dims[d])
            end
            mod_factor = exp(im * 2π * phase)
            
            # Accumulate
            src.data[i] += sub_grid[i] * mod_factor
        end
    end
    
    # Note: Normalization handled by solver (weights).
end


function build_recombination_map(spec_asu::SpectralIndexing, real_asu::CrystallographicASU, direct_ops::Vector{SymOp}, all_blocks::Vector{ASUBlock}, L_factors::Vector{Vector{Int}})
    
    N = spec_asu.N
    dim = length(N)
    
    # Detect Mode B: all blocks have orbit info and same L factor
    is_mode_b = !isnothing(all_blocks[1].orbit)
    
    if !is_mode_b
        return _build_recombination_map_mode_a(spec_asu, real_asu, direct_ops, all_blocks, L_factors)
    end
    
    # ===== Mode B: DIT with Orbit Reduction =====
    # Formula: F(h) = Σ_{x₀ ∈ X₀} exp(-2πi h·x₀/N) · Y_{x₀}(h mod M)
    # With orbit reduction:
    #   Y_{y₀}(q) = exp(-2πi q·m/M) · Y_{rep}(R_g^T q mod M)
    #   where m = (R_g·rep + t_g - y₀) / L
    
    L = L_factors[1]  # All blocks share same L
    M = [N[d] ÷ L[d] for d in 1:dim]  # Subgrid size
    
    # Build orbit lookup: x₀ → (block_index, R_g, m_vec)
    # Each block corresponds to an orbit representative
    block_reps = Dict{Vector{Int}, Int}()  # rep → block_index
    for (b_i, block) in enumerate(all_blocks)
        rep = block.orbit.representative
        block_reps[rep] = b_i
    end
    
    # For each x₀ in the full coarse grid, find the orbit rep and mapping operation
    all_shifts = vec(collect(Iterators.product([0:L[d]-1 for d in 1:dim]...)))
    
    shift_info = Dict{Vector{Int}, Tuple{Int, Matrix{Int}, Vector{Int}}}()  # x₀ → (block_idx, R_g, m_vec)
    
    for block in all_blocks
        orbit = block.orbit
        rep = orbit.representative
        b_i = block_reps[rep]
        
        for (i, member) in enumerate(orbit.members)
            if member == rep
                # Identity: maps rep to itself
                shift_info[member] = (b_i, Matrix{Int}(I, dim, dim), zeros(Int, dim))
            else
                # Find operation g: R_g·rep + t_g ≡ member (mod L)
                for op in orbit.ops[i]
                    x_prime = mod.(op.R * rep .+ op.t, L)
                    if x_prime == member
                        m_vec = (op.R * rep .+ Int.(op.t) .- member) .÷ L
                        shift_info[member] = (b_i, op.R, m_vec)
                        break
                    end
                end
            end
        end
    end
    
    # Register active blocks (one per orbit rep, r=0)
    active_lookup = Dict{Int, Int}()  # block_idx → buffer_offset
    active_blocks = Vector{ActiveBlock}()
    current_offset = 1
    
    for (b_i, block) in enumerate(all_blocks)
        dims = size(block.data)
        len = length(block.data)
        active_lookup[b_i] = current_offset
        push!(active_blocks, ActiveBlock(b_i, zeros(Int, dim), current_offset))
        current_offset += len
    end
    
    # Build sparse recombination matrix
    I_idx = Vector{Int}()
    J_idx = Vector{Int}()
    V_val = Vector{ComplexF64}()
    
    for (h_idx, pt) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        h_local = [mod(h_vec[d], M[d]) for d in 1:dim]  # 0-based local frequency
        
        # Accumulate contributions from all L^D shifts
        for x0_tuple in all_shifts
            x0 = collect(x0_tuple)
            
            if !haskey(shift_info, x0)
                continue
            end
            
            b_i, R_g, m_vec = shift_info[x0]
            
            # 1. Twiddle factor: exp(-2πi h·x₀/N)
            twiddle_phase = sum(h_vec .* x0 ./ N)
            twiddle = exp(-im * 2π * twiddle_phase)
            
            # 2. Phase correction from orbit reduction: exp(-2πi h_local·m/M)
            correction_phase = sum(h_local .* m_vec ./ M)
            correction = exp(-im * 2π * correction_phase)
            
            # 3. Rotated local frequency: R_g^T · h_local mod M
            rot_h = mod.(transpose(R_g) * h_local, M)
            
            # Linear index in buffer for this block
            lin_idx = 1
            stride = 1
            for d in 1:dim
                lin_idx += rot_h[d] * stride
                stride *= M[d]
            end
            
            col_idx = active_lookup[b_i] + (lin_idx - 1)
            weight = twiddle * correction
            
            push!(I_idx, h_idx)
            push!(J_idx, col_idx)
            push!(V_val, weight)
        end
    end
    
    total_size = current_offset - 1
    n_orbits = length(all_blocks)
    n_subgrids = prod(L)
    println("DEBUG [build_recombination_map Mode B]: n_orbits=$n_orbits/$n_subgrids subgrids, buffer_size=$total_size, speedup=$(n_subgrids/n_orbits)x")
    
    M_sparse = sparse(I_idx, J_idx, V_val, length(spec_asu.points), total_size)
    return M_sparse, active_blocks, total_size
end

# Mode A recombination (original code)
function _build_recombination_map_mode_a(spec_asu::SpectralIndexing, real_asu::CrystallographicASU, direct_ops::Vector{SymOp}, all_blocks::Vector{ASUBlock}, L_factors::Vector{Vector{Int}})
    
    N = spec_asu.N
    dim = length(N)
    
    active_lookup = Dict{Tuple{Int, Vector{Int}}, Int}()
    active_blocks = Vector{ActiveBlock}()
    current_offset = 1
    
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{ComplexF64}()
    
    for (h_idx, pt) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        
        for g in direct_ops
            q_vec = transpose(g.R) * h_vec
            
            phase_val = 0.0
            for d in 1:dim
                 phase_val += h_vec[d] * g.t[d] / N[d]
            end
            weight = exp(-im * 2π * phase_val)
            
            shift_phase = 0.0
            for d in 1:dim
                shift_phase += h_vec[d] * real_asu.shift[d] / N[d]
            end
            weight *= exp(-im * 2π * shift_phase)
            
            for (b_i, block) in enumerate(all_blocks)
                L = L_factors[b_i]
                dims = size(block.data)
                
                x_0 = [first(r) for r in block.range]
                phase_b = 0.0
                for d in 1:dim
                    phase_b += q_vec[d] * x_0[d] / N[d]
                end
                weight_b = exp(-im * 2π * phase_b)
                
                k_local = [mod(q_vec[d], dims[d]) for d in 1:dim]
                
                lin_idx_local = 1
                stride = 1
                for d in 1:dim
                    lin_idx_local += k_local[d] * stride
                    stride *= dims[d]
                end
                
                r_vec = zeros(Int, dim)
                key = (b_i, r_vec)
                if !haskey(active_lookup, key)
                    len = length(block.data)
                    active_lookup[key] = current_offset
                    push!(active_blocks, ActiveBlock(b_i, r_vec, current_offset))
                    current_offset += len
                end
                buffer_base = active_lookup[key]
                col_idx = buffer_base + (lin_idx_local - 1)
                
                total_weight = weight * weight_b
                
                push!(I, h_idx)
                push!(J, col_idx)
                push!(V, total_weight)
            end
        end
    end
    
    total_size = current_offset - 1
    println("DEBUG [build_recombination_map Mode A]: n_ops=$(length(direct_ops)), buffer_size=$total_size, active_blocks=$(length(active_blocks))")
    
    M = sparse(I, J, V, length(spec_asu.points), total_size)
    return M, active_blocks, total_size
end

# Dummy flat buffers
function flatten_to_buffer!(buffer::AbstractVector, asu::CrystallographicASU)
end
function unflatten_from_buffer!(asu::CrystallographicASU, buffer::AbstractVector)
end

include("recursive_blocks.jl")

end
