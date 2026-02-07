
using Pkg
Pkg.activate(".")
using Test
using LinearAlgebra
using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.DiffusionSolver
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using CrystallographicFFT.SpectralIndexing
using Crystalline
using FFTW
using Random

# --- Helpers ---

"""
    expand_to_full_grid(asu, N, ops)

Expands the ASU data to the full N-dimensional grid using symmetry operations.
"""
function expand_to_full_grid(asu::CrystallographicASU{D, T, A}, N::Tuple, ops) where {D, T, A}
    full_grid = zeros(ComplexF64, N)
    
    # We need to correctly handle the "multiplicity" or "weighting" if we just want to verify the FIELD.
    # The ASU stores the field value u(r). This value exists at all symmetry equivalent points.
    # So we just copy u(r) to all r' = op * r.
    
    # Iterate over all blocks in the ASU
    for (d, blocks) in asu.dim_blocks
        for block in blocks
            # Iterate over all points in the block
            cis = CartesianIndices(size(block.data))
            for index in cis
                val = block.data[index]
                
                # Get the global 0-based index of this point
                # r_idx_0based = [start + step*i for ...] 
                # block.range contains absolute 0-based indices for the full grid?
                # Yes, asu.jl documentation/implementation implies ranges are global indices.
                
                r_idx_0based = [block.range[k][index[k]] for k in 1:D]
                
                # Apply all symmetry operations to generate the orbit
                # Note: We must be careful not to double-write if points are special.
                # However, for expansion, we just write the value.
                # If multiple ops map to the same point, they should map the SAME value (scalar field symmetry).
                
                # Generate orbit
                orbit = Set{Vector{Int}}()
                # Start with identity (or just current point)
                push!(orbit, r_idx_0based)
                
                # Simple orbit expansion
                stack = [r_idx_0based]
                while !isempty(stack)
                    curr = pop!(stack)
                    for op in ops
                        # Apply Op: x_new = W*x + w
                        next_p = apply_op(op, curr, N)
                        if !(next_p in orbit)
                            push!(orbit, next_p)
                            push!(stack, next_p)
                        end
                    end
                end
                
                # Write to full grid
                for p in orbit
                    # 1-based indexing for Julia Array
                    full_grid[(p .+ 1)...] = val
                end
            end
        end
    end
    return full_grid
end



@testset "KRFFT vs FFTW Verification" begin
    # Setup p2mm (SG 5 in 2D)
    N = (16, 16)
    lattice = Matrix(I*1.0, 2, 2)
    dim = 2
    sg_num = 5 # p2mm
    Δs = 0.01

    println("Planning KRFFT Solver...")
    solver = plan_diffusion(N, lattice, sg_num, dim, Δs; method=:krfft)
    
    # Get Operations for Expansion
    # We can fetch them from Crystalline or the solver
    # solver.plan.recombination_map construction uses `direct_ops`.
    # But for full grid expansion we need the full space group ops.
    ops = get_ops(sg_num, dim, N) # From helper or standard lib
    # Helper to convert Crystalline ops to SymOps if needed, but `expand_to_full_grid` needs our SymOp or adaptable.
    # Let's use internal `solver.real_asu` context if possible, but actually we just need the list of SymOps.
    # We can use `CrystallographicFFT.SymmetryOps.generate_symops(sg_num, dim, N)` if available, or just parse Crystalline.
    # `plan_diffusion` calls `generate_symops`.
    
    # Quick fix: Use the same ops as used in planning.
    # But `plan_diffusion` doesn't store them all publicly easily?
    # Actually `CrystallographicFFT.SymmetryOps.get_ops` is not exported.
    # Let's rely on `Crystalline` direct.
    
    # Get Operations using internal API
    # Must use Shifted Ops for correct Magic Shift Expansion!
    ops_base = CrystallographicFFT.SymmetryOps.get_ops(sg_num, dim, N)
    shift, shifted_ops = CrystallographicFFT.ASU.find_optimal_shift(ops_base, N)
    ops = shifted_ops
    
    # 1. Random Real Field on ASU
    println("Initializing Random Field...")
    # Manual fill
    for (d, blocks) in solver.real_asu.dim_blocks
        for b in blocks
            b.data .= rand(ComplexF64, size(b.data))
        end
    end
    
    # 2. Expand to Full Grid (Reference)
    println("Expanding to Full Grid...")
    # expand_to_full_grid uses apply_op.
    # We need to make sure expand_to_full_grid calls the correct apply_op.
    # Since we imported CrystallographicFFT.SymmetryOps, apply_op is available?
    # No, we defined a local apply_op in this file. We should delete it.
    u_full = expand_to_full_grid(solver.real_asu, N, ops)
    
    # 3. FFTW Reference
    println("Running FFTW...")
    u_spec_full = fft(u_full) # Standard FFT
    
    # 4. KRFFT Forward
    println("Running KRFFT Forward...")
    # Re-zero buffers
    fill!(solver.plan.work_buffer, 0.0)
    fill!(solver.u_spec_buf, 0.0)
    
    # Manual Forward Step w/ Weights
    # A. Apply Weights
    # Note: weights are scaling factors for integration.
    # If standard FFT effectively integrates sum(f(x)), it implicitly has weight 1.
    # Our `apply_scale!` adds multiplicity weights.
    # Does KRFFT assume input is "Weighted Field" or "Field"?
    # The `apply_diffusion_operator!` calls `apply_scale!`.
    # So the input to `map_fft!` is indeed "Weighted Field".
    CrystallographicFFT.DiffusionSolver.apply_scale!(solver.real_asu, solver.weights)
    
    # B. FFT Blocks
    map_fft!(solver.plan, solver.real_asu)
    
    # C. Flatten & Recombine
    # Note: map_fft! updates work_buffer
    # flatten_to_buffer! is deprecated/no-op in modulated strategy.
    
    # Check if we messed up vector vs array?
    # work_buffer is Complex Buffer.
    # recombination_map expects what?
    # Modulated build_recombination_map creates M with `total_size` cols.
    # `plan.work_buffer` is size `buffer_size`.
    # They match.
    # But `work_buffer` is 1D array in our implementation?
    # `GeneralCFFTPlan` line 208: `work_buffer = zeros(ComplexF64, buffer_size)`.
    # So it is Vector.
    
    mul!(solver.u_spec_buf, solver.plan.recombination_map, solver.plan.work_buffer)
    
    # 5. Compare
    println("Comparing Spectral Coefficients...")
    u_spec_krfft = solver.u_spec_buf
    spec_asu = solver.spec_asu
    
    max_diff = 0.0
    
    # Collect diffs
    diffs = Float64[]
    
    for (i, pt) in enumerate(spec_asu.points)
        k_vec = get_k_vector(spec_asu, i)
        
        # Map k to 1-based index
        # FFTW output has standard layout: 0, 1, ..., N/2, -N/2+1, ... -1
        # Julia arrays are 1-based.
        # k=0 -> 1
        # k=1 -> 2
        # k=-1 -> N
        
        k_idx = [mod(k, N[d]) + 1 for (d, k) in enumerate(k_vec)]
        
        val_ref = u_spec_full[k_idx...]
        val_krfft = u_spec_krfft[i]
        
        # Normalize Reference
        val_ref_norm = val_ref / prod(N)
        
        # Apply Magic Shift Correction to Reference to match KRFFT
        # KRFFT output is exp(-i k . s) * DFT
        phase_shift = 0.0
        for d in 1:length(N)
            # k_vec is integer index
            phase_shift += k_vec[d] * solver.real_asu.shift[d] / N[d]
        end
        val_ref_norm *= exp(-im * 2π * phase_shift) # Re-enable check
        
        diff = abs(val_krfft - val_ref_norm)
        push!(diffs, diff)
        
        if diff > max_diff
            max_diff = diff
            if diff > 1e-10
                 println("Mismatch at k=$k_vec: KRFFT=$val_krfft, RefNorm=$val_ref_norm, Diff=$diff")
            end
        end
    end
    
    println("Max Spectral Difference: $max_diff")
    println("Mean Spectral Difference: $(sum(diffs)/length(diffs))")
    
    @test max_diff < 1e-10
    
end
