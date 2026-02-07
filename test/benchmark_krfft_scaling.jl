
using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.SpectralIndexing
using CrystallographicFFT.KRFFT
using FFTW
using Printf
using Statistics
using LinearAlgebra

function run_benchmark()
    # Use Pmmm (Space Group 47) for |G|=8 (max reduction)
    # Fair comparison: FFTW single-threaded, in-place plan
    
    FFTW.set_num_threads(1)
    
    sizes = [(32,32,32), (64,64,64)]
    
    println("Benchmark: Pmmm (|G|=8) Mode B vs Full Grid FFTW")
    println("N             | FFTW (ms) | Mode B (ms) | Speedup | Orbits | FFTs")
    println("--------------|-----------|-------------|---------|--------|------")
    
    for N in sizes
        # Pmmm (47) has 8 operations
        ops = get_ops(47, 3, N)
        n_ops = length(ops)
        
        u_full = rand(ComplexF64, N)
        
        # 1. FFTW Baseline - Fair: plan + mul!
        fftw_plan = plan_fft!(copy(u_full); flags=FFTW.MEASURE)
        work = copy(u_full)
        
        t_fftw_samples = Float64[]
        for _ in 1:10
            copyto!(work, u_full)
            t = @elapsed mul!(work, fftw_plan, work)
            push!(t_fftw_samples, t)
        end
        t_fftw = minimum(t_fftw_samples)
        
        # 2. Mode B
        real_asu = pack_asu_interleaved(u_full, N, ops; L=(2,2,2))
        n_orbits = length(real_asu.dim_blocks[3])  # Number of orbit reps
        
        spec_asu = calc_spectral_asu(ops, 3, N)
        plan = plan_krfft(real_asu, spec_asu, ops)
        n_active = length(plan.active_blocks)
        
        function execute_mode_b(p, asu)
            map_fft!(p, asu)
            return p.recombination_map * p.work_buffer
        end
        
        # Warmup
        execute_mode_b(plan, real_asu)
        
        # Profile: map_fft! time
        t_map_fft_samples = Float64[]
        for _ in 1:10
            t = @elapsed map_fft!(plan, real_asu)
            push!(t_map_fft_samples, t)
        end
        t_map_fft = minimum(t_map_fft_samples)
        
        # Profile: recombination time
        t_recomb_samples = Float64[]
        for _ in 1:10
            t = @elapsed (plan.recombination_map * plan.work_buffer)
            push!(t_recomb_samples, t)
        end
        t_recomb = minimum(t_recomb_samples)
        
        t_mode_b_samples = Float64[]
        for _ in 1:10
            t = @elapsed execute_mode_b(plan, real_asu)
            push!(t_mode_b_samples, t)
        end
        t_mode_b = minimum(t_mode_b_samples)
        
        speedup = t_fftw / t_mode_b
        
        @printf("%-13s | %9.3f | %11.3f | %7.2f | %6d | %4d\n", 
                string(N), t_fftw*1000, t_mode_b*1000, speedup, n_orbits, n_active)
        @printf("  (Profile: map_fft=%.3f ms, recomb=%.3f ms, |G|=%d)\n", 
                t_map_fft*1000, t_recomb*1000, n_ops)
    end
end

run_benchmark()
