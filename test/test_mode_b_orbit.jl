"""
Benchmark: KRFFT L=2 optimized pipeline for SCFT.
Measures FFT + reconstruct (no pack — data pre-stored in input_buffer).
"""

using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT
using FFTW
using LinearAlgebra

FFTW.set_num_threads(1)

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

function benchmark(sg, N, L)
    dim = length(N)
    N_vec = collect(N)
    
    ops_orig = get_ops(sg, dim, N)
    shift, ops_shifted = find_optimal_shift(ops_orig, N)
    
    println("\n" * "="^60)
    println("SG $sg, N=$N, L=$L, |G|=$(length(ops_orig))")
    println("="^60)
    
    u = rand(Float64, N)
    symmetrize_shifted!(u, ops_shifted, N_vec)
    F_ref = fft(u)
    
    # Build plan
    asu = pack_asu_interleaved(u, N, ops_shifted; L=L, asu_only=true)
    spec = calc_spectral_asu(ops_shifted, dim, N)
    plan = plan_krfft(asu, spec, ops_shifted)
    
    M_sub = plan.subgrid_dims
    n_spec = length(spec.points)
    println("Subgrid: $(Tuple(M_sub)), spec_asu: $n_spec")
    
    # Extract subgrid
    sub = complex.(u[1:2:end, 1:2:end, 1:2:end])
    
    # === Correctness: SCFT path ===
    plan.input_buffer .= vec(sub)
    F_calc = fft_reconstruct!(plan)
    
    max_err = 0.0
    for (h_idx, _) in enumerate(spec.points)
        h = get_k_vector(spec, h_idx)
        h_1based = CartesianIndex(Tuple(mod.(h, N_vec) .+ 1)...)
        err = abs(F_calc[h_idx] - F_ref[h_1based])
        max_err = max(max_err, err)
    end
    println("Spectral error: $max_err")
    println("PASS: $(max_err < 1e-8)")
    
    # === Timing ===
    nreps = 200
    
    # Baseline: FFTW N³ plan + mul!
    u_c = complex(u)
    F_full = similar(u_c)
    p_full = plan_fft(u_c)
    mul!(F_full, p_full, u_c)
    t_fftw = @elapsed for _ in 1:nreps; mul!(F_full, p_full, u_c); end
    t_fftw /= nreps
    
    # FFT-only subgrid: plan + mul!
    F_sub = similar(sub)
    p_sub = plan_fft(sub)
    mul!(F_sub, p_sub, sub)
    t_fft_sub = @elapsed for _ in 1:nreps; mul!(F_sub, p_sub, sub); end
    t_fft_sub /= nreps
    
    # SCFT path: fft_reconstruct! (out-of-place FFT + reconstruct)
    plan.input_buffer .= vec(sub)
    fft_reconstruct!(plan)  # warmup
    t_pipeline = @elapsed for _ in 1:nreps
        plan.input_buffer .= vec(sub)
        fft_reconstruct!(plan)
    end
    t_pipeline /= nreps
    
    # Component: reconstruct only
    mul!(F_sub, p_sub, sub)
    plan.work_buffer .= vec(F_sub)
    fast_reconstruct!(plan)  # warmup
    t_recon = @elapsed for _ in 1:nreps
        fast_reconstruct!(plan)
    end
    t_recon /= nreps
    
    println("\nTiming (μs):")
    println("  Baseline (FFTW N³):    $(round(t_fftw*1e6, digits=1))")
    println("  ─── Components ───")
    println("  FFT sub (plan+mul!):   $(round(t_fft_sub*1e6, digits=1))")
    println("  Reconstruct:           $(round(t_recon*1e6, digits=1))")
    println("  ─── SCFT path ───")
    println("  fft_reconstruct!:      $(round(t_pipeline*1e6, digits=1))")
    println("\nSpeedup:")
    println("  FFT-only (M³ vs N³):   $(round(t_fftw/t_fft_sub, digits=2))x")
    println("  SCFT pipeline:         $(round(t_fftw/t_pipeline, digits=2))x")
    println("  Theoretical ceiling:   $(length(ops_orig))x")
    println("  Recon/FFT ratio:       $(round(t_recon/t_fft_sub, digits=2))")
end

println("KRFFT L=2: Pmmm SCFT Benchmark (out-of-place FFT)")
println("=" ^ 55)

benchmark(47, (16, 16, 16), (2, 2, 2))
benchmark(47, (32, 32, 32), (2, 2, 2))
benchmark(47, (64, 64, 64), (2, 2, 2))
benchmark(47, (128, 128, 128), (2, 2, 2))
