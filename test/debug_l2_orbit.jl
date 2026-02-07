"""
Test: Mode A with asu_only=true, L=2, shifted ops for Pmmm.
Data symmetrized with shifted ops.
"""

using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT
using FFTW
using LinearAlgebra

function symmetrize_shifted!(u, ops, N_vec)
    u_copy = copy(u)
    fill!(u, 0.0)
    for i in CartesianIndices(u)
        idx = collect(Tuple(i) .- 1)
        val = 0.0
        for op in ops
            new_idx = mod.(op.R * idx .+ op.t, N_vec)
            new_i = CartesianIndex(Tuple(new_idx .+ 1)...)
            val += u_copy[new_i]
        end
        u[i] = val / length(ops)
    end
end

function test_mode_a_l2()
    N = (16, 16, 16)
    L = (2, 2, 2)
    dim = 3
    N_vec = collect(N)
    
    ops_orig = get_ops(47, dim, N)
    shift, ops_shifted = find_optimal_shift(ops_orig, N)
    println("Pmmm N=$N, L=$L, |G|=$(length(ops_orig))")
    println("shift = $shift")
    
    # 使用 shifted ops 对称化
    u = rand(Float64, N)
    symmetrize_shifted!(u, ops_shifted, N_vec)
    F_ref = fft(u)
    
    # Mode A: asu_only=true, 用 shifted ops
    asu = pack_asu_interleaved(u, N, ops_shifted; L=L, asu_only=true)
    
    all_blocks = Vector{ASUBlock}()
    for d in sort(collect(keys(asu.dim_blocks)))
        append!(all_blocks, asu.dim_blocks[d])
    end
    println("Blocks: $(length(all_blocks)), Size: $(size(all_blocks[1].data))")
    println("Block range: $([b.range for b in all_blocks])")
    println("Block orbit: $(all_blocks[1].orbit)")
    
    spec = calc_spectral_asu(ops_shifted, dim, N)
    println("Spectral ASU points: $(length(spec.points))")
    
    plan = plan_krfft(asu, spec, ops_shifted)
    map_fft!(plan, asu)
    F_calc = plan.recombination_map * plan.work_buffer
    
    max_err = 0.0
    for (h_idx, _) in enumerate(spec.points)
        h = get_k_vector(spec, h_idx)
        h_1based = CartesianIndex(Tuple(mod.(h, N_vec) .+ 1)...)
        err = abs(F_calc[h_idx] - F_ref[h_1based])
        if err > max_err
            max_err = err
        end
    end
    
    println("\nMax spectral error: $max_err")
    println("PASS: $(max_err < 1e-8)")
    
    if max_err < 1e-8
        # Performance
        t_fftw = @elapsed for _ in 1:100; fft(u); end; t_fftw /= 100
        asu_f = pack_asu_interleaved(u, N, ops_shifted; L=L, asu_only=true)
        t_fft_only = @elapsed for _ in 1:100
            map_fft!(plan, asu_f)
        end
        t_fft_only /= 100
        
        println("\nFull FFT:     $(round(t_fftw*1e6, digits=1)) μs")
        println("ASU FFT:      $(round(t_fft_only*1e6, digits=1)) μs")
        println("FFT speedup:  $(round(t_fftw/t_fft_only, digits=2))x")
        println("Theoretical:  $(length(ops_orig))x")
    end
end

test_mode_a_l2()
