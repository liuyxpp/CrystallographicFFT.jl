
using Pkg
Pkg.activate(".")
using Test
using LinearAlgebra
using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.DiffusionSolver
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using Crystalline

@testset "General KRFFT vs Matrix Solver" begin
    # 1. Setup p2mm (SG 5 in 2D)
    N = (16, 16)
    lattice = Matrix(I*1.0, 2, 2)
    dim = 2
    sg_num = 5 # p2mm
    Δs = 0.01

    println("Planning Matrix Solver...")
    solver_mat = plan_diffusion(N, lattice, sg_num, dim, Δs; method=:matrix)
    
    println("Planning KRFFT Solver...")
    solver_krfft = try
        plan_diffusion(N, lattice, sg_num, dim, Δs; method=:krfft)
    catch e
        println("Error in plan_diffusion (KRFFT):")
        showerror(stdout, e, catch_backtrace())
        println()
        rethrow(e)
    end
    
    # 2. Random Initial Condition (Same for both)
    n_pts = length(solver_mat.real_asu)
    u0_vec = rand(ComplexF64, n_pts) .+ 0.0
    
    # Set Matrix Solver State
    u_mat = copy(u0_vec)
    w_mat = ones(n_pts) * 0.1
    
    # Fill KRFFT with Constant 1.0 for Debugging Logic
    u_krfft = solver_krfft.real_asu
    w_krfft = CrystallographicFFT.ASU.pack_asu(solver_mat.real_asu, N, ComplexF64; shift=solver_krfft.real_asu.shift) 
    
    shift = u_krfft.shift
    
    fill_func(r_idx) = 1.0 + 0.0im # Constant Field
    
    # Identify K=0 index
    k0_idx = 0
    for (i, k_pt) in enumerate(solver_mat.spec_asu.points)
        k_vec = CrystallographicFFT.SpectralIndexing.get_k_vector(solver_mat.spec_asu, i)
        if all(k_vec .== 0)
            k0_idx = i
            println("Found k=0 at index $i")
            break
        end
    end
    if k0_idx == 0
        println("WARNING: k=0 not found in spec_asu!")
        # Proceed assuming index 1?? No, print error.
    end
    
    # Fill KRFFT
    for dim_g in values(u_krfft.dim_blocks)
        for b in dim_g
            iter = Iterators.product(b.range...)
            for (local_idx, r_idx_tuple) in enumerate(iter)
                # r_idx_tuple is 0-based index from range
                b.data[local_idx] = fill_func(collect(r_idx_tuple))
            end
        end
    end
    
    # Fill w_krfft with 0.1
    for dim_g in values(w_krfft.dim_blocks)
        for b in dim_g
            fill!(b.data, 0.1 + 0.0im)
        end
    end
    
    # Fill Matrix Vector
    for (i, p) in enumerate(solver_mat.real_asu)
        u_mat[i] = fill_func(p.idx)
    end
    
    # 3. Step Diffusion
    println("Stepping...")
    step_diffusion!(solver_mat, u_mat, w_mat, Δs)
    step_diffusion!(solver_krfft, u_krfft, w_krfft, Δs)
    
    # =========================================================================
    # Stepwise Verification
    # =========================================================================
    println("\n--- Stepwise Verification ---")
    
    # 1. Forward Transform Verification
    println("\n[Step 1] Verifying Forward Transform (Real -> Spec)...")
    
    # Run Matrix Forward
    u_spec_mat = solver_mat.M_inv * u_mat
    
    # Run KRFFT Forward
    # Re-initialize buffer to be safe
    fill!(solver_krfft.plan.work_buffer, 0.0)
    fill!(solver_krfft.u_spec_buf, 0.0)
    
    # Execute Forward Parts manually
    # Debug Weights
    w_vals = []
    for d in values(solver_krfft.weights.dim_blocks)
        for b in d
            append!(w_vals, vec(b.data))
        end
    end
    println("Weights Mean: $(sum(w_vals)/length(w_vals))")
    println("Weights Extremadata: $(minimum(w_vals)) - $(maximum(w_vals))")
    
    # RESET u_real to 1.0 to ensure clean Forward Check independent of step_diffusion result
    for dim_g in values(solver_krfft.real_asu.dim_blocks)
        for b in dim_g
            fill!(b.data, 1.0 + 0.0im)
        end
    end
    
    # 1. Apply Weights
    CrystallographicFFT.DiffusionSolver.apply_scale!(solver_krfft.real_asu, solver_krfft.weights)
    
    # 2. FFT
    map_fft!(solver_krfft.real_asu)
    flatten_to_buffer!(solver_krfft.plan.work_buffer, solver_krfft.real_asu)
    mul!(solver_krfft.u_spec_buf, solver_krfft.plan.recombination_map, solver_krfft.plan.work_buffer)
    u_spec_krfft = solver_krfft.u_spec_buf
    
    # Restore u_real to unweighted/untransformed for Step 2?
    # Actually u_spec_krfft is what we compare. u_real is consumed.
    # Note: solver_krfft.real_asu is modified.
    # We should probably reset it or unscale it if we re-use.
    # But for Step 1 check we end here.
    # Step 2 uses u_spec_krfft (copied).
    # Step 3 uses u_spec_diff_krfft.
    # So u_real state doesn't matter unless Step 3 uses it?
    # Step 3 Overwrites u_real via `unflatten`. So we are fine.
    
    # Compare
    if k0_idx > 0
        println("u_spec_mat[k0]: $(u_spec_mat[k0_idx])")
        println("u_spec_krfft[k0]: $(u_spec_krfft[k0_idx])")
    end
    
    diff_fwd = norm(u_spec_mat - u_spec_krfft) / norm(u_spec_mat)
    
    if diff_fwd > 1e-10
         println("Forward Check FAILED. Diff: $diff_fwd")
    else
         println("Forward Check PASSED.")
    end
    # Ensure this is reached
    @test diff_fwd < 1e-10
    
    # 2. Diffusion Step Verification (Spectral Space)
    println("\n[Step 2] Verifying Diffusion Operator (Spec -> Spec)...")
    # Matrix: Q * u_spec
    u_spec_diff_mat = solver_mat.Q * u_spec_mat
    
    # KRFFT: Q * u_spec (In-place)
    # We use a copy to verify logic without mutating the buffer for Step 3 yet
    u_spec_diff_krfft = copy(u_spec_krfft)
    lmul!(solver_krfft.Q, u_spec_diff_krfft)
    
    diff_diff = norm(u_spec_diff_mat - u_spec_diff_krfft) / norm(u_spec_diff_mat)
    println("Diffusion Relative Error: $diff_diff")
    @test diff_diff < 1e-10
    
    # 3. Inverse Transform Verification
    println("\n[Step 3] Verifying Inverse Transform (Spec -> Real)...")
    
    # Run Matrix Inverse: M * u_spec_diff
    u_real_final_mat = solver_mat.M * u_spec_diff_mat
    # Note: MatrixSolver applies real() at the end usually.
    
    # Run KRFFT Inverse manually
    # Load diffuse spec back to buffer
    copyto!(solver_krfft.u_spec_buf, u_spec_diff_krfft) 
    
    # Adjoint Map
    # u_blocks = M' * u_spec
    mul!(solver_krfft.plan.work_buffer, adjoint(solver_krfft.plan.recombination_map), solver_krfft.u_spec_buf)
    
    # Scatter
    unflatten_from_buffer!(solver_krfft.real_asu, solver_krfft.plan.work_buffer)

    # Manual Apply Inverse Scaling / Normalization (as implemented in apply_diffusion_operator!)
    # We must replicate the logic in apply_diffusion_operator! here to test it.
    # Logic: b.data .*= scale_factor; ifft!
    
    # But wait, apply_diffusion_operator! does: unflatten -> ifft -> norm.
    # Wait, my apply_diffusion_operator! implements:
    # 1. mul! (adjoint) -> buffer
    # 2. unflatten -> blocks
    # 3. map_ifft! -> blocks
    # 4. normalization -> blocks
    
    # BUT `map_ifft!` expects data in Frequency domain?
    # NO. `unflatten` puts data into blocks.
    # The Adjoint Map `M'` produces "Aliased Block FFT Coefficients" if M was "Un-aliasing".
    # Yes. M maps Block Coeffs -> Global Coeffs.
    # M' maps Global Coeffs -> Sum of Block Coeffs copies.
    # So the buffer holds Frequency domain data for blocks.
    # So `unflatten` fills `block.data` with Frequency data.
    # Then `map_ifft!` transforms Freq -> Real. Correct.
    
    # Replicate normalization loop (Inverse Recovery)
    N_total = prod(N)
    for d_k in keys(u_krfft.dim_blocks)
        for (i, b) in enumerate(u_krfft.dim_blocks[d_k])
             w_b = solver_krfft.weights.dim_blocks[d_k][i]
             
             # Combined Scale: (8 * N_total) / weight
             scale_factor = 8 * N_total
             
             @. b.data = b.data * scale_factor / w_b.data
             
             @. b.data = complex(real(b.data), 0.0)
        end
    end
    
    # IFFT
    map_ifft!(u_krfft)
    
    # Compare Final Real Points
    max_err_inv = 0.0
    for dim_g in values(u_krfft.dim_blocks)
        for b in dim_g
             iter = Iterators.product(b.range...)
             for (local_idx, r_idx_tuple) in enumerate(iter)
                 r_idx = collect(r_idx_tuple)
                 val_krfft = b.data[local_idx]
                 
                 found = false
                 for (i, p) in enumerate(solver_mat.real_asu)
                     if p.idx == r_idx
                         val_mat = u_real_final_mat[i]
                         # Complex vs Real? Matrix solver returns Complex before 'real()' call in apply 
                         # But let's check magnitude
                         err = abs(val_krfft - val_mat)
                         if err > max_err_inv
                             max_err_inv = err
                         end
                         found = true
                         break
                     end
                 end
                 if !found
                     error("Point $r_idx not found")
                 end
             end
        end
    end
    
    println("Inverse Reconstruction Max Error: $max_err_inv")
    @test max_err_inv < 1e-10

end
