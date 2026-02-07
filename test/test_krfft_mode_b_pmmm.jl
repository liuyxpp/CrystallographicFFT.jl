"""
Test KRFFT Mode B with Pmmm (Space Group 47) for |G|=8 speedup verification.

This test validates the correct implementation of KRFFT V Eq. 27:
    F(h) = Σ_g e_A(h, t_g) * Y(R_g^T h_1)

Where:
- Y is the FFT of single ASU subgrid Γ₀
- The sum over g ∈ G reconstructs all contributions using symmetry-phase formula
"""

using Test
using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using CrystallographicFFT.SpectralIndexing
using FFTW

@testset "KRFFT Mode B - Pmmm (|G|=8)" begin
    N = (16, 16, 16)
    ops = get_ops(47, 3, N)  # Pmmm |G|=8
    L = (2, 2, 2)
    
    @testset "Structure Verification" begin
        # Create test data
        u = rand(Float64, N)
        
        # Pack ASU (Mode B, asu_only=true)
        real_asu = pack_asu_interleaved(u, N, ops; L=L, asu_only=true)
        
        # Verify single block
        @test haskey(real_asu.dim_blocks, 3)
        @test length(real_asu.dim_blocks[3]) == 1
        
        block = real_asu.dim_blocks[3][1]
        expected_block_size = Tuple(N[d] ÷ L[d] for d in 1:3)
        @test size(block.data) == expected_block_size
        
        # Verify KRFFT plan
        spec_asu = calc_spectral_asu(ops, 3, N)
        plan = plan_krfft(real_asu, spec_asu, ops)
        
        # Verify 8x reduction: buffer = N/8
        @test length(plan.active_blocks) == 1
        expected_buffer = prod(expected_block_size)
        @test length(plan.work_buffer) == expected_buffer
        
        println("✓ Buffer size: $(length(plan.work_buffer)) (expected: $expected_buffer)")
        println("✓ Active blocks: $(length(plan.active_blocks)) (expected: 1)")
    end
    
    @testset "Spectral Correctness" begin
        N = (16, 16, 16)
        ops = get_ops(47, 3, N)
        L = (2, 2, 2)
        
        # Create Pmmm-symmetric test data by symmetrizing random input
        # Pmmm: invariant under (x→-x), (y→-y), (z→-z) reflections
        u_raw = rand(Float64, N)
        u = zeros(Float64, N)
        
        # Symmetrize under Pmmm operations
        for i in CartesianIndices(u)
            idx = Tuple(i) .- 1  # 0-based
            val = 0.0
            for op in ops
                # Apply operation: idx' = R * idx mod N
                new_idx = mod.(op.R * collect(idx), collect(N))
                new_i = CartesianIndex(Tuple(new_idx .+ 1)...)
                val += u_raw[new_i]
            end
            u[i] = val / length(ops)
        end
        
        # FFTW reference
        fft_ref = fft(u)
        
        # Mode B KRFFT
        real_asu = pack_asu_interleaved(u, N, ops; L=L, asu_only=true)
        spec_asu = calc_spectral_asu(ops, 3, N)
        plan = plan_krfft(real_asu, spec_asu, ops)
        
        # Execute Forward FFT
        map_fft!(plan, real_asu)
        
        # Recombine
        result = plan.recombination_map * plan.work_buffer
        
        # Compare with FFTW at spectral ASU points
        max_err = 0.0
        for (h_idx, pt) in enumerate(spec_asu.points)
            h = get_k_vector(spec_asu, h_idx)
            
            # Convert to 1-based FFTW indexing
            fftw_idx = Tuple(mod(h[d], N[d]) + 1 for d in 1:3)
            ref_val = fft_ref[fftw_idx...]
            calc_val = result[h_idx]
            
            err = abs(calc_val - ref_val)
            max_err = max(max_err, err)
            
            if err > 1e-10
                println("h=$h: calc=$calc_val, ref=$ref_val, err=$err")
            end
        end
        
        println("Max spectral error: $max_err")
        @test max_err < 1e-10
    end
end
