# KRFFT Performance Benchmark
# Compares full-grid FFT vs KRFFT (subgrid FFT + reconstruct) for diag/gen paths.
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.ASU
using CrystallographicFFT.SpectralIndexing
using CrystallographicFFT.KRFFT
using FFTW
using Statistics
using LinearAlgebra: mul!

function make_symmetric_field(ops, N)
    u = rand(N...)
    u_sym = zeros(N...)
    for op in ops
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(op.R * x .+ op.t, collect(N)) .+ 1
            u_sym[idx] += u[x2...]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

function benchmark_sg(sg, N_tuple; n_warmup=3, n_trials=20)
    dim = length(N_tuple)
    ops = get_ops(sg, dim, N_tuple)
    _, ops_s = find_optimal_shift(ops, N_tuple)
    
    L = auto_L(ops_s)
    spec = calc_spectral_asu(ops_s, dim, N_tuple)
    plan = plan_krfft(spec, ops_s)
    
    path = length(plan.phase_factors) == 3 && 
           length(plan.output_buffer) == prod(plan.subgrid_dims) ? "diag" : "gen "
    
    # Create symmetric test field
    u_sym = make_symmetric_field(ops_s, N_tuple)
    u_complex = complex(u_sym)
    
    # === Benchmark full-grid FFT ===
    fft_plan = plan_fft(u_complex)
    F_full = similar(u_complex)
    
    # Warmup
    for _ in 1:n_warmup
        mul!(F_full, fft_plan, u_complex)
    end
    
    # Measure
    t_fft = Float64[]
    for _ in 1:n_trials
        t = @elapsed mul!(F_full, fft_plan, u_complex)
        push!(t_fft, t)
    end
    
    # === Benchmark KRFFT (extract + FFT + reconstruct) ===
    f0 = u_sym[1:L[1]:end, 1:L[2]:end, 1:L[3]:end]
    
    # Warmup
    for _ in 1:n_warmup
        plan.input_buffer .= vec(complex(f0))
        fft_reconstruct!(plan)
    end
    
    # Measure (pack + fft + recon together)
    t_krfft = Float64[]
    for _ in 1:n_trials
        t = @elapsed begin
            plan.input_buffer .= vec(complex(f0))
            fft_reconstruct!(plan)
        end
        push!(t_krfft, t)
    end
    
    # Also measure reconstruct only
    plan.input_buffer .= vec(complex(f0))
    sub_in = reshape(plan.input_buffer, Tuple(plan.subgrid_dims))
    sub_out = reshape(plan.work_buffer, Tuple(plan.subgrid_dims))
    mul!(sub_out, plan.sub_plans[1], sub_in)
    
    t_recon = Float64[]
    for _ in 1:n_trials
        t = @elapsed fast_reconstruct!(plan)
        push!(t_recon, t)
    end
    
    # Compute stats (use median for stability)
    fft_ms = median(t_fft) * 1000
    krfft_ms = median(t_krfft) * 1000
    recon_ms = median(t_recon) * 1000
    subfft_ms = krfft_ms - recon_ms  # approximate
    speedup = fft_ms / krfft_ms
    
    n_spec = length(spec.points)
    n_full = prod(N_tuple)
    asu_ratio = n_spec / n_full
    
    return (sg=sg, G=length(ops_s), L=L, path=path,
            fft_ms=fft_ms, krfft_ms=krfft_ms, 
            subfft_ms=subfft_ms, recon_ms=recon_ms,
            speedup=speedup, n_spec=n_spec, n_full=n_full,
            asu_ratio=asu_ratio, prodL=prod(L))
end

function run_benchmarks()
    # Test groups covering both paths
    test_groups = [
        # Orthorhombic (diag path)
        # (47, "Pmmm"),
        # (25, "Pmm2"),
        # (6, "Pm"),
        # High-symmetry diag
        # (123, "P4/mmm"),
        # (221, "Pm-3m"),
        # (229, "Im-3m"),
        (225, "Fm-3m"),
        (223, "Pm-3n"),
        # Gen path (glide/screw)
        (136, "P42/mnm"),
        (70, "Fddd"),
        (224, "Pn-3m"),
        (230, "Ia-3d"),
    ]
    
    for N_size in [128]
        N = (N_size, N_size, N_size)
        println("\n" * "="^85)
        println("Grid N = $N")
        println("="^85)
        println(rpad("SG", 6) * rpad("Name", 10) * rpad("|G|", 5) * 
                rpad("L", 12) * rpad("Path", 6) * 
                rpad("FFT(ms)", 10) * rpad("KRFFT(ms)", 11) *
                rpad("Recon(ms)", 11) * rpad("Speedup", 8) *
                rpad("ASU%", 6))
        println("-"^85)
        
        for (sg, name) in test_groups
            try
                r = benchmark_sg(sg, N; n_warmup=5, n_trials=30)
                println(rpad(string(r.sg), 6) * 
                        rpad(name, 10) *
                        rpad(string(r.G), 5) *
                        rpad(string(r.L), 12) *
                        rpad(r.path, 6) *
                        rpad(string(round(r.fft_ms, digits=2)), 10) *
                        rpad(string(round(r.krfft_ms, digits=3)), 11) *
                        rpad(string(round(r.recon_ms, digits=3)), 11) *
                        rpad(string(round(r.speedup, digits=2)) * "x", 8) *
                        rpad(string(round(r.asu_ratio * 100, digits=1)) * "%", 6))
                GC.gc()
            catch e
                println(rpad(string(sg), 6) * rpad(name, 10) * "ERROR: $e")
            end
        end
    end
end

run_benchmarks()
