
using Test
using CrystallographicFFT
using CrystallographicFFT.ASU # ASUBlock, CrystallographicASU
# using CrystallographicFFT.DataStructs # Not found
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.SpectralIndexing
using CrystallographicFFT.KRFFT
using FFTW
using LinearAlgebra
using Random

@testset "KRFFT Mode B Integration" begin
    # 1. Setup P4 Symmetry
    N = (16, 16, 16)
    ops = get_ops(75, 3, N) # P4
    
    # Create input signal with P4 symmetry
    # Use random signal project to symmetry
    Random.seed!(42)
    u_full = rand(ComplexF64, N)
    u_sym = zeros(ComplexF64, N)
    
    # Simple symmetrization
    for op in ops
        # Inefficient but correct reference
        # x' = R x + t
        # u(x) = u(x')
        # Average over group
        # Actually we need u_sym(x) = sum u(g x)
        # Or just enforce u(g x) = u(x)
        # Let's use `symmetrize_field!` from Crystalline if available, 
        # or just construct a symmetric function manually.
        # f(x,y,z) = cos(2pi x) + cos(2pi y) + ...
    end
    
    # Analytic symmetric function:
    # P4: (x,y,z) -> (-y,x,z)
    # f = cos(2pi x/N) * cos(2pi y/N) * cos(2pi z/N)
    # This is symmetric under x<->y (partially) and x->-y.
    # Actually P4 requires 4-fold.
    # f = cos(2pi x) + cos(2pi y) is P4 invariant.
    
    for i in CartesianIndices(N)
        x, y, z = Tuple(i) .- 1
        # P4 invariant term
        val = cos(2π * x / 16) + cos(2π * y / 16) + cos(2π * z / 16) * 0.5
        # Wait, cos(x)+cos(y) is P4?
        # x->-y: cos(-y)+cos(x) = cos(y)+cos(x). Yes.
        u_sym[i] = val
    end
    
    # 2. Pack Mode B
    real_asu = pack_asu_interleaved(u_sym, N, ops; L=(2,2,2))
    
    # Verify blocks count
    # P4 with L=2 should have 6 blocks.
    @test length(real_asu.dim_blocks[3]) == 6
    
    # 3. Spectral Indexing
    # Calculate Spectral ASU based on symmetry
    # P4 (75) for reciprocal space
    # We need reciprocal operations. 
    # calc_spectral_asu handles dual_ops internally.
    spec_asu = calc_spectral_asu(ops, 3, N)
    
    # 4. Plan KRFFT
    plan = plan_krfft(real_asu, spec_asu, ops)
    
    # 5. Execute
    map_fft!(plan, real_asu)
    
    # 6. Verify against FFTW
    fft_ref = fft(u_sym)
    
    # Check specific HK points
    # Since we computed all, `plan.work_buffer` has coefficients mapped by `recombination_map`.
    # result = M * work_buffer
    
    result_spectrum = plan.recombination_map * plan.work_buffer
    
    # Compare with reference at indices in `spec_asu.points`
    # spec_asu.points stores (h, k, l).
    # We need to find linear index in FFTW output.
    
    max_err = 0.0
    # KRFFT is normalized (1/N), FFTW is unnormalized.
    # We must scale KRFFT result by prod(N) to match FFTW.
    normalization_factor = prod(N)
    
    for (i, pt) in enumerate(spec_asu.points)
        h = pt.idx
        # FFTW index: h (mod N) + 1
        # pt.idx are in 0..N-1 range from ASUPoint
        idx = [mod(h[d], N[d]) + 1 for d in 1:3]
        ref_val = fft_ref[idx...]
        
        # Scale up
        calc_val = result_spectrum[i] * normalization_factor
        
        err = abs(calc_val - ref_val)
        if err > max_err
            max_err = err
            println("Error at h=$h: calc=$calc_val, ref=$ref_val")
        end
    end
    
    println("Max Spectral Error (Mode B): ", max_err)
    if max_err > 1e-10
         println("FAILING TEST WITH $max_err")
    end
    @test max_err < 1e-10
end
