module KRFFT

using KernelAbstractions
using AbstractFFTs
using FFTW
using LinearAlgebra
using SparseArrays
using ..ASU: CrystallographicASU, ASUBlock
using ..ASU
using ..SpectralIndexing: SpectralIndexing, get_k_vector
using ..SymmetryOps: SymOp, apply_op!, CenteringType, CentP, CentC, CentA, CentI, CentF
using ..SymmetryOps: detect_centering_type

export GeneralCFFTPlan, map_fft!, map_ifft!, build_recombination_map, plan_krfft
export fast_reconstruct!, pack_stride!, execute_krfft!, ReconEntry, fft_reconstruct!
export auto_L
export plan_krfft_recursive, execute_recursive_krfft!, pack_p3c!, fft_p3c!
export fill_g0_butterfly!, reconstruct_from_g0!
export plan_krfft_sparse, execute_sparse_krfft!, sparse_reconstruct!, detect_centering
export plan_krfft_selective, execute_selective_krfft!
export plan_krfft_g0asu, execute_g0asu_krfft!
export plan_krfft_g0asu_general, execute_general_g0asu_krfft!, GeneralG0ASUPlan
export plan_krfft_g0asu_backward, execute_g0asu_ikrfft!
export FractalNode, FractalCFFTPlan, calc_asu_tree, build_recursive_tree
export collect_leaves, collect_inner_nodes_bottomup, tree_summary
export plan_fractal_krfft, execute_fractal_krfft!
export OptimizedFractalPlan, plan_fractal_krfft_v2, execute_fractal_krfft_v2!
export SubgridCenteringFoldPlan, CenteredKRFFTPlan
export plan_krfft_centered, execute_centered_krfft!, fft_reconstruct_centered!
export plan_centering_fold, centering_fold!, fft_channels!, assemble_G0!
export ifft_channels!, centering_unfold!, disassemble_G0!
export CenteredKRFFTBackwardPlan, InvReconEntry
export plan_centered_ikrfft, execute_centered_ikrfft!, ifft_unrecon_centered!
export CenteredSCFTPlan, plan_centered_scft, execute_centered_scft!, update_kernel!
export M2BackwardPlan, plan_m2_backward, execute_m2_backward!


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

# ============================================================================
# M2 Backward Transform (Inverse of General KRFFT)
# ============================================================================
#
# Mathematical basis (per-fiber inverse butterfly):
#   Forward: F̂(h) = Σ_a w_a(h) · Y₀(R_aᵀ h mod M)
#   Backward: Y₀(q) = Σ_a B⁻¹[0,a](q) · F_full(h_a)
#   where h_a are the d full-grid frequencies in fiber q.
#
# Since n_active=1, we only need B⁻¹'s first row per fiber.

"""
    InvReconEntry

Precomputed inverse reconstruction info for one (subgrid_freq, fiber_member) pair.
Combines B⁻¹ coefficient with spectral ASU mapping phase.
"""
struct InvReconEntry
    spec_idx::Int32        # Index into spectral ASU (1-based)
    weight::ComplexF64     # Combined B⁻¹ coefficient × symmetry/Hermitian phase
    conj_flag::Bool        # If true, conjugate F_spec[spec_idx] before multiplying
end

"""
    M2BackwardPlan

Plan for the M2 backward transform (inverse of General KRFFT).
Precomputed inverse reconstruction table enables efficient spectral ASU → subgrid IFFT.
"""
struct M2BackwardPlan
    # Inverse reconstruction table: (d, prod(M))
    inv_recon_table::Matrix{InvReconEntry}
    d::Int                                   # fiber length = prod(L)

    # IFFT
    ifft_plan::Any                           # FFTW plan for M-grid
    Y_buf::Vector{ComplexF64}                # M³ complex buffer (inv_recon output / IFFT input)
    f0_buf::Vector{ComplexF64}               # M³ complex buffer (IFFT output)

    # Grid metadata
    L::Vector{Int}
    subgrid_dims::Vector{Int}                # M = N/L
    grid_N::Vector{Int}                      # N

    # Pmmm separable fast path data
    is_separable::Bool
    inv_phase_factors::Vector{Vector{ComplexF64}}  # 1D inverse twiddle factors
end

"""
    plan_m2_backward(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp}) -> M2BackwardPlan

Construct the M2 backward plan by building the inverse reconstruction table.

For each subgrid frequency q ∈ [0,M)^D:
1. Enumerate the d full-grid frequencies in the fiber
2. Build the d×d butterfly matrix B(q)
3. Compute B⁻¹'s first row
4. Map each full-grid frequency to spectral ASU (via symmetry/Hermitian)
5. Store combined weight = B⁻¹ coeff × symmetry phase
"""
function plan_m2_backward(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    N_vec = collect(N)

    # 1. Auto-determine L and M (same as forward)
    L = auto_L(ops_shifted)
    M_sub = [N[d] ÷ L[d] for d in 1:dim]

    if any(L .* M_sub .!= N_vec)
        error("Grid size N=$N not divisible by auto L=$L.")
    end

    # 2. Select representative ops per subgrid (same logic as plan_krfft)
    subgrid_reps = Dict{Vector{Int}, SymOp}()
    subgrid_quality = Dict{Vector{Int}, Int}()

    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], L[d]) for d in 1:dim]
        is_diag = all(op.R[i,j] == 0 for i in 1:dim for j in 1:dim if i != j)
        simple_t = all(mod(t[d], N[d]) ∈ (0, N[d]-1) for d in 1:dim)
        quality = is_diag ? (simple_t ? 2 : 1) : 0
        if !haskey(subgrid_reps, x0) || quality > subgrid_quality[x0]
            subgrid_reps[x0] = op
            subgrid_quality[x0] = quality
        end
    end

    d = prod(L)
    rep_ops = Vector{SymOp}(undef, d)
    sub_idx = 0
    for x0 in Iterators.product([0:L[dd]-1 for dd in 1:dim]...)
        sub_idx += 1
        x0_vec = collect(x0)
        if haskey(subgrid_reps, x0_vec)
            rep_ops[sub_idx] = subgrid_reps[x0_vec]
        else
            error("Subgrid x₀=$x0_vec not reachable. auto_L should have prevented this.")
        end
    end

    # 3. Build spectral reverse lookup: full-grid h → (spec_idx, phase, conj_flag)
    #    For each full-grid frequency, find which spectral ASU point represents it
    h_to_spec = _build_spectral_reverse_lookup(spec_asu, ops_shifted, N_vec, dim)

    # 4. Build inverse reconstruction table
    M_vol = prod(M_sub)
    inv_recon_table = Matrix{InvReconEntry}(undef, d, M_vol)

    # Compute subgrid parities (α vectors) for each rep_op
    alphas = Vector{Vector{Int}}(undef, d)
    a_idx = 0
    for x0 in Iterators.product([0:L[dd]-1 for dd in 1:dim]...)
        a_idx += 1
        alphas[a_idx] = collect(x0)
    end

    # Pre-allocate working arrays
    B_matrix = zeros(ComplexF64, d, d)
    h_vecs = [zeros(Int, dim) for _ in 1:d]  # full-grid frequencies per fiber

    for q_cart in CartesianIndices(Tuple(M_sub))
        q_vec = [q_cart[dd] - 1 for dd in 1:dim]  # 0-based
        q_lin = LinearIndices(Tuple(M_sub))[q_cart]

        # 4a. Enumerate full-grid frequencies in this fiber
        for a in 1:d
            for dd in 1:dim
                h_vecs[a][dd] = q_vec[dd] + M_sub[dd] * alphas[a][dd]
            end
        end

        # 4b. Build d×d butterfly matrix B(q)
        #     B[a, b] = exp(-2πi h_a · t_b / N) × δ(R_bᵀ h_a mod M maps correctly)
        #     But simpler: B[a, b] = weight of rep_op b applied to h_a
        fill!(B_matrix, zero(ComplexF64))
        for a in 1:d
            h = h_vecs[a]
            for b in 1:d
                g = rep_ops[b]
                # Phase: exp(-2πi h · t_g / N)
                phase_val = 0.0
                for dd in 1:dim
                    phase_val += h[dd] * g.t[dd] / N[dd]
                end
                w = cispi(-2 * phase_val)

                # Rotated frequency → subgrid index
                rot_q_lin = 0
                stride = 1
                for dd in 1:dim
                    rot_val = 0
                    for dd2 in 1:dim
                        rot_val += g.R[dd2, dd] * h[dd2]
                    end
                    rot_val = mod(rot_val, M_sub[dd])
                    rot_q_lin += rot_val * stride
                    stride *= M_sub[dd]
                end
                # B[a, b] contributes to the position rot_q_lin in Y₀
                # For the standard M2 case, all rep_ops map the same q to different
                # or same positions. The correct B construction is:
                # F(h_a) = Σ_b B[a,b] · Y₀(gather[b])
                # where gather[b] = R_bᵀ h_a mod M → linear index
                B_matrix[a, b] = w
            end
        end

        # 4c. Compute B⁻¹ first row
        B_inv_row1 = inv(B_matrix)[1, :]

        # 4d. Map full-grid frequencies to spectral ASU and build entries
        for a in 1:d
            h = h_vecs[a]
            h_mod = [mod(h[dd], N_vec[dd]) for dd in 1:dim]

            if haskey(h_to_spec, h_mod)
                spec_idx, sym_phase, conj_f = h_to_spec[h_mod]
                combined_weight = B_inv_row1[a] * sym_phase
                inv_recon_table[a, q_lin] = InvReconEntry(
                    Int32(spec_idx), combined_weight, conj_f
                )
            else
                # Extinguished frequency: F(h) = 0, contributes nothing
                inv_recon_table[a, q_lin] = InvReconEntry(
                    Int32(1), zero(ComplexF64), false
                )
            end
        end
    end

    # 5. Plan IFFT
    dummy = zeros(ComplexF64, Tuple(M_sub))
    ifft_plan = plan_ifft(dummy)

    # 6. Allocate buffers
    Y_buf = zeros(ComplexF64, M_vol)
    f0_buf = zeros(ComplexF64, M_vol)

    # 7. Detect Pmmm separable structure
    all_diagonal = all(op -> all(op.R[i,j] == 0 for i in 1:dim for j in 1:dim if i != j), rep_ops)
    simple_translations = all(op -> all(mod(round(Int, op.t[dd]), N[dd]) ∈ (0, N[dd]-1)
                                        for dd in 1:dim), rep_ops)
    is_sep = all_diagonal && simple_translations && d == 2^dim

    inv_phase_factors = Vector{ComplexF64}[]
    if is_sep
        # For Pmmm inverse: phase factor for dimension d at freq q is
        # conj(φ_d(q)) / 2 where φ_d(q) = exp(2πi q / N_d)
        inv_phase_factors = Vector{Vector{ComplexF64}}(undef, dim)
        for dd in 1:dim
            inv_phase_factors[dd] = [cispi(-2 * q / N[dd]) for q in 0:M_sub[dd]-1]
        end
    end

    return M2BackwardPlan(
        inv_recon_table, d,
        ifft_plan, Y_buf, f0_buf,
        L, M_sub, N_vec,
        is_sep, inv_phase_factors
    )
end


"""
    _build_spectral_reverse_lookup(spec_asu, ops_shifted, N_vec, dim)

Build a dictionary mapping every full-grid frequency h ∈ [0,N)^D to its
spectral ASU representative: (spec_idx, phase, conj_flag).

For h in the spectral ASU: direct mapping.
For h related by symmetry: find g such that R_g^T h ≡ h_asu (mod N).
For h related by Hermitian symmetry: map -h → h_asu with conjugation.
"""
function _build_spectral_reverse_lookup(spec_asu::SpectralIndexing,
                                        ops_shifted::Vector{SymOp},
                                        N_vec::Vector{Int}, dim::Int)
    # Result: h_mod (0-based) → (spec_idx, phase, conj_flag)
    h_to_spec = Dict{Vector{Int}, Tuple{Int, ComplexF64, Bool}}()

    n_spec = length(spec_asu.points)

    # Reciprocal-space ops: R* = (R⁻¹)ᵀ = R for orthogonal ops, but use dual_ops
    # Actually, the forward recon uses R^T (transpose of the direct-space rotation).
    # The spectral ASU uses dual_ops (reciprocal ops).
    # For reverse lookup: we need to map R_g^T h back to h_asu.
    # The direct_ops have R_direct, and in the recon: R_directᵀ · h mod M.
    # For spectral orbit: h and R_recip · h are equivalent
    # where R_recip comes from dual_ops.
    recip_ops = spec_asu.ops  # Already the reciprocal-space ops stored in SpectralIndexing

    # Build fast index lookup for spectral ASU points
    spec_point_map = Dict{Vector{Int}, Int}()
    for (idx, pt) in enumerate(spec_asu.points)
        spec_point_map[pt.idx] = idx
    end

    # For each spectral ASU point, expand its orbit
    for (spec_idx, pt) in enumerate(spec_asu.points)
        h_asu = pt.idx  # 0-based

        # Identity: h_asu itself
        if !haskey(h_to_spec, h_asu)
            h_to_spec[h_asu] = (spec_idx, complex(1.0), false)
        end

        # Apply all direct-space ops to get full-grid equivalents
        # Spectral symmetry (numerically verified):
        #   F(R^T h) = exp(+2πi h·t/N) · F(h)
        # So: F_full(h') = exp(+2πi h_asu·t/N) · F_spec(spec_idx)
        # where h' = R^T · h_asu mod N
        for op in ops_shifted
            R = op.R
            t = op.t
            # h' = R^T · h_asu mod N
            h_rot = zeros(Int, dim)
            for dd in 1:dim
                s = 0
                for dd2 in 1:dim
                    s += R[dd2, dd] * h_asu[dd2]
                end
                h_rot[dd] = mod(s, N_vec[dd])
            end

            if !haskey(h_to_spec, h_rot)
                # F_full(h') = exp(+2πi h_asu · t / N) · F_spec
                phase_val = 0.0
                for dd in 1:dim
                    phase_val += h_asu[dd] * t[dd] / N_vec[dd]
                end
                phase = cispi(2 * phase_val)
                h_to_spec[h_rot] = (spec_idx, phase, false)
            end
        end

        # Hermitian symmetry: F(-h) = conj(F(h)) for real-valued fields
        # So F_full(-h_asu mod N) = conj(F_spec(spec_idx))
        h_neg = [mod(-h_asu[dd], N_vec[dd]) for dd in 1:dim]
        if !haskey(h_to_spec, h_neg)
            h_to_spec[h_neg] = (spec_idx, complex(1.0), true)
        end

        # Also: Hermitian of orbit members
        # F_full(R^T h) = exp(+2πi h·t/N) · F_spec
        # F_full(-R^T h) = conj(F_full(R^T h)) = exp(-2πi h·t/N) · conj(F_spec)
        # phase for conj = exp(+2πi h·t/N) conj'd = exp(-2πi h·t/N)
        for op in ops_shifted
            R = op.R
            t = op.t
            # h' = R^T · h_asu mod N
            h_rot = zeros(Int, dim)
            for dd in 1:dim
                s = 0
                for dd2 in 1:dim
                    s += R[dd2, dd] * h_asu[dd2]
                end
                h_rot[dd] = mod(s, N_vec[dd])
            end

            h_rot_neg = [mod(-h_rot[dd], N_vec[dd]) for dd in 1:dim]
            if !haskey(h_to_spec, h_rot_neg)
                # F_full(-h') = conj(F_full(h')) = exp(-2πi h_asu·t/N) · conj(F_spec)
                phase_val = 0.0
                for dd in 1:dim
                    phase_val += h_asu[dd] * t[dd] / N_vec[dd]
                end
                phase = cispi(-2 * phase_val)
                h_to_spec[h_rot_neg] = (spec_idx, phase, true)
            end
        end
    end

    return h_to_spec
end


"""
    execute_m2_backward!(bplan::M2BackwardPlan, F_spec::AbstractVector{ComplexF64}) -> Vector{ComplexF64}

Execute the M2 backward transform: spectral ASU → subgrid (M³).

Steps:
1. Inverse reconstruction: gather from F_spec into Y₀ using inv_recon_table
2. IFFT: Y₀(M³) → f₀(M³)

Returns `bplan.f0_buf` containing the subgrid real-space data.
"""
function execute_m2_backward!(bplan::M2BackwardPlan, F_spec::AbstractVector{ComplexF64})
    # Step 1: Inverse reconstruction
    _inv_reconstruct_m2!(bplan, F_spec)

    # Step 2: IFFT
    Y_in = reshape(bplan.Y_buf, Tuple(bplan.subgrid_dims))
    f_out = reshape(bplan.f0_buf, Tuple(bplan.subgrid_dims))
    mul!(f_out, bplan.ifft_plan, Y_in)  # Apply IFFT: Y₀ → f₀

    return bplan.f0_buf
end

"""
General table-based inverse reconstruction: spectral ASU → Y₀(M³).
"""
function _inv_reconstruct_m2!(bplan::M2BackwardPlan, F_spec::AbstractVector{ComplexF64})
    d = bplan.d
    M_vol = prod(bplan.subgrid_dims)
    Y = bplan.Y_buf
    table = bplan.inv_recon_table

    @inbounds for q_lin in 1:M_vol
        val = zero(ComplexF64)
        for a in 1:d
            entry = table[a, q_lin]
            f_val = entry.conj_flag ? conj(F_spec[entry.spec_idx]) : F_spec[entry.spec_idx]
            val += entry.weight * f_val
        end
        Y[q_lin] = val
    end
end


include("recursive_blocks.jl")
include("recursive_blocks_backward.jl")
include("fractal_krfft.jl")
include("fractal_krfft_v3.jl")
include("centering_prefold.jl")
include("centering_fold.jl")


end
