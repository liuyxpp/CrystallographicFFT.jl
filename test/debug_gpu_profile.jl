# Detailed per-step profiling of centered CFFT pipeline on CPU vs GPU.
# Profiles Fm-3m (SG 225, F-centered, 4 channels) at N=128.
#
using CrystallographicFFT, CUDA, CUDA.CUFFT, Printf
using CrystallographicFFT: centering_fold!, fft_channels!, assemble_G0!,
    _reconstruct_general!, _reconstruct_pmmm!, fft_reconstruct_centered!,
    execute_centered_backward!, centering_unfold!, disassemble_G0!,
    ifft_channels!, _inv_recon_orbit!

println("GPU: ", CUDA.name(CUDA.device()))

# ── Setup ─────────────────────────────────────────────────────────────────────

N = (128, 128, 128)
sg = 225  # Fm-3m

# CPU plan
pair_cpu = plan_cfft_pair(N, sg, 3; array_type=Array{Float64})
M_cpu = pair_cpu.M
f0_cpu = rand(Float64, M_cpu...)
F_cpu = zeros(ComplexF64, pair_cpu.n_spec)

# GPU plan
pair_gpu = plan_cfft_pair(N, sg, 3; array_type=CuArray{Float64})
M_gpu = pair_gpu.M
f0_gpu = CuArray(f0_cpu)
F_gpu = CuArray{ComplexF64}(undef, pair_gpu.n_spec)

println("M = $(M_cpu), n_spec = $(pair_cpu.n_spec)")
println()

# ── CPU profiling ─────────────────────────────────────────────────────────────

function profile_cpu_fwd(pair, f0, F; trials=20)
    fwd = pair.fwd
    krfft = fwd.krfft_plan
    fold = fwd.fold_plan

    # warmup
    cfft!(F, pair, f0)

    t_copy_in = Inf; t_fold = Inf; t_fft = Inf; t_assemble = Inf; t_recon = Inf; t_copy_out = Inf; t_total = Inf

    for _ in 1:trials
        # copy in
        t0 = time_ns()
        @inbounds for k in 1:size(f0,3), j in 1:size(f0,2), i in 1:size(f0,1)
            fwd.f0_buffer[i,j,k] = f0[i,j,k]
        end
        t_copy_in = min(t_copy_in, (time_ns()-t0)/1e3)

        # centering fold
        t0 = time_ns()
        centering_fold!(fold, fwd.f0_buffer)
        t_fold = min(t_fold, (time_ns()-t0)/1e3)

        # fft channels
        t0 = time_ns()
        fft_channels!(fold)
        t_fft = min(t_fft, (time_ns()-t0)/1e3)

        # assemble G0
        t0 = time_ns()
        assemble_G0!(fwd.G0_view, fold)
        t_assemble = min(t_assemble, (time_ns()-t0)/1e3)

        # reconstruct
        t0 = time_ns()
        M_vol = prod(krfft.M)
        if krfft.is_pmmm && krfft.n_spec == M_vol && length(krfft.phase_factors) == length(krfft.M)
            _reconstruct_pmmm!(krfft)
        else
            _reconstruct_general!(krfft)
        end
        t_recon = min(t_recon, (time_ns()-t0)/1e3)

        # copy out
        t0 = time_ns()
        @inbounds @simd for i in 1:length(F)
            F[i] = krfft.output_buffer[i]
        end
        t_copy_out = min(t_copy_out, (time_ns()-t0)/1e3)

        # total
        t0 = time_ns()
        cfft!(F, pair, f0)
        t_total = min(t_total, (time_ns()-t0)/1e3)
    end

    return (copy_in=t_copy_in, fold=t_fold, fft=t_fft, assemble=t_assemble,
            recon=t_recon, copy_out=t_copy_out, total=t_total)
end

# ── GPU profiling ─────────────────────────────────────────────────────────────

function profile_gpu_fwd(pair, f0, F; trials=20)
    fwd = pair.fwd
    krfft = fwd.krfft_plan
    fold = fwd.fold_plan

    # warmup
    cfft!(F, pair, f0)
    CUDA.synchronize()

    t_copy_in = Inf; t_fold = Inf; t_fft = Inf; t_assemble = Inf; t_recon = Inf; t_copy_out = Inf; t_total = Inf

    for _ in 1:trials
        # copy in
        CUDA.synchronize()
        t0 = time_ns()
        copyto!(fwd.f0_buffer, f0)
        CUDA.synchronize()
        t_copy_in = min(t_copy_in, (time_ns()-t0)/1e3)

        # centering fold
        CUDA.synchronize()
        t0 = time_ns()
        centering_fold!(fold, fwd.f0_buffer)
        CUDA.synchronize()
        t_fold = min(t_fold, (time_ns()-t0)/1e3)

        # fft channels (batched CUFFT)
        CUDA.synchronize()
        t0 = time_ns()
        fft_channels!(fold)
        CUDA.synchronize()
        t_fft = min(t_fft, (time_ns()-t0)/1e3)

        # assemble G0 (fused)
        CUDA.synchronize()
        t0 = time_ns()
        assemble_G0!(fwd.G0_view, fold)
        CUDA.synchronize()
        t_assemble = min(t_assemble, (time_ns()-t0)/1e3)

        # reconstruct
        CUDA.synchronize()
        t0 = time_ns()
        _reconstruct_general!(krfft)
        CUDA.synchronize()
        t_recon = min(t_recon, (time_ns()-t0)/1e3)

        # copy out
        CUDA.synchronize()
        t0 = time_ns()
        copyto!(F, krfft.output_buffer)
        CUDA.synchronize()
        t_copy_out = min(t_copy_out, (time_ns()-t0)/1e3)

        # total (end-to-end)
        CUDA.synchronize()
        t0 = time_ns()
        cfft!(F, pair, f0)
        CUDA.synchronize()
        t_total = min(t_total, (time_ns()-t0)/1e3)
    end

    return (copy_in=t_copy_in, fold=t_fold, fft=t_fft, assemble=t_assemble,
            recon=t_recon, copy_out=t_copy_out, total=t_total)
end

# ── Run ───────────────────────────────────────────────────────────────────────

cpu = profile_cpu_fwd(pair_cpu, f0_cpu, F_cpu)
gpu = profile_gpu_fwd(pair_gpu, f0_gpu, F_gpu)

# CUFFT baseline
g = CUDA.rand(ComplexF64, N...)
p = CUFFT.plan_fft(g)
p * g; CUDA.synchronize()
t_cufft = let ts=Float64[]; for _ in 1:20; CUDA.synchronize(); t0=time_ns(); p*g; CUDA.synchronize(); push!(ts,(time_ns()-t0)/1e3); end; minimum(ts); end

# ── Print ─────────────────────────────────────────────────────────────────────

println("=" ^ 90)
println("Fm-3m (SG 225) Centered Forward — Per-Step Breakdown (μs)")
println("=" ^ 90)
println()
steps = [:copy_in, :fold, :fft, :assemble, :recon, :copy_out, :total]
labels = ["1. copy f0 → buffer", "2. centering fold", "3. FFT channels",
          "4. assemble G₀", "5. reconstruct", "6. copy out F̂", "── TOTAL ──"]

@printf("%-25s %10s %10s %10s %10s\n", "Step", "CPU (μs)", "GPU (μs)", "GPU/CPU", "GPU %")
println("-" ^ 70)
for (s, lab) in zip(steps, labels)
    c = getfield(cpu, s)
    g = getfield(gpu, s)
    ratio = g / c
    pct = g / gpu.total * 100
    if s == :total
        println("-" ^ 70)
    end
    @printf("%-25s %10.1f %10.1f %10.2f %9.1f%%\n", lab, c, g, ratio, pct)
end
println()
println("CUFFT(128³) baseline = $(round(t_cufft/1e3, digits=3)) ms")
println("CPU  speedup vs CUFFT(128³): $(round(t_cufft / cpu.total, digits=2))×")
println("GPU  speedup vs CUFFT(128³): $(round(t_cufft / gpu.total, digits=2))×")
println("GPU parity = $(round(gpu.total > 0 ? (t_cufft/gpu.total) / (t_cufft/cpu.total) : 0, digits=3))")
println()

# Sum of parts vs total
parts_cpu = cpu.copy_in + cpu.fold + cpu.fft + cpu.assemble + cpu.recon + cpu.copy_out
parts_gpu = gpu.copy_in + gpu.fold + gpu.fft + gpu.assemble + gpu.recon + gpu.copy_out
println("Sum-of-parts: CPU=$(round(parts_cpu, digits=1))μs  GPU=$(round(parts_gpu, digits=1))μs")
println("Overhead (total - parts): CPU=$(round(cpu.total - parts_cpu, digits=1))μs  GPU=$(round(gpu.total - parts_gpu, digits=1))μs")

# Also profile Pm-3m general for comparison
println("\n" * "=" ^ 90)
println("Pm-3m (SG 225) General Forward — Per-Step Breakdown (μs)")
println("=" ^ 90)

pair_gen_cpu = plan_cfft_pair(N, 200, 3; array_type=Array{Float64})  # Pm-3 = SG200, general path
pair_gen_gpu = plan_cfft_pair(N, 200, 3; array_type=CuArray{Float64})
f0g_cpu = rand(Float64, pair_gen_cpu.M...)
Fg_cpu = zeros(ComplexF64, pair_gen_cpu.n_spec)
f0g_gpu = CuArray(f0g_cpu)
Fg_gpu = CuArray{ComplexF64}(undef, pair_gen_gpu.n_spec)

function profile_general_fwd(pair, f0, F, is_gpu; trials=20)
    fwd = pair.fwd
    cfft!(F, pair, f0)
    if is_gpu; CUDA.synchronize(); end

    t_copy_in = Inf; t_fft = Inf; t_recon = Inf; t_copy_out = Inf; t_total = Inf

    for _ in 1:trials
        if is_gpu; CUDA.synchronize(); end

        # copy real → complex
        if is_gpu; CUDA.synchronize(); end
        t0 = time_ns()
        M_vol = prod(fwd.M)
        if is_gpu
            CrystallographicFFT.copy_real_to_complex_kernel!(CUDABackend())(
                fwd.input_buffer, f0; ndrange=M_vol)
            CUDA.synchronize()
        else
            @inbounds @simd for i in 1:M_vol
                fwd.input_buffer[i] = complex(f0[i])
            end
        end
        t_copy_in = min(t_copy_in, (time_ns()-t0)/1e3)

        # FFT
        if is_gpu; CUDA.synchronize(); end
        t0 = time_ns()
        mul!(fwd.work_buffer, fwd.fft_plan, fwd.input_buffer)
        if is_gpu; CUDA.synchronize(); end
        t_fft = min(t_fft, (time_ns()-t0)/1e3)

        # reconstruct
        if is_gpu; CUDA.synchronize(); end
        t0 = time_ns()
        _reconstruct_general!(fwd)
        if is_gpu; CUDA.synchronize(); end
        t_recon = min(t_recon, (time_ns()-t0)/1e3)

        # copy out
        if is_gpu; CUDA.synchronize(); end
        t0 = time_ns()
        if is_gpu
            copyto!(F, fwd.output_buffer)
            CUDA.synchronize()
        else
            @inbounds @simd for i in 1:length(F)
                F[i] = fwd.output_buffer[i]
            end
        end
        t_copy_out = min(t_copy_out, (time_ns()-t0)/1e3)

        # total
        if is_gpu; CUDA.synchronize(); end
        t0 = time_ns()
        cfft!(F, pair, f0)
        if is_gpu; CUDA.synchronize(); end
        t_total = min(t_total, (time_ns()-t0)/1e3)
    end

    return (copy_in=t_copy_in, fft=t_fft, recon=t_recon, copy_out=t_copy_out, total=t_total)
end

gcpu = profile_general_fwd(pair_gen_cpu, f0g_cpu, Fg_cpu, false)
ggpu = profile_general_fwd(pair_gen_gpu, f0g_gpu, Fg_gpu, true)

println()
steps_g = [:copy_in, :fft, :recon, :copy_out, :total]
labels_g = ["1. copy real→complex", "2. FFT(M³)", "3. reconstruct", "4. copy out F̂", "── TOTAL ──"]

@printf("%-25s %10s %10s %10s %10s\n", "Step", "CPU (μs)", "GPU (μs)", "GPU/CPU", "GPU %")
println("-" ^ 70)
for (s, lab) in zip(steps_g, labels_g)
    c = getfield(gcpu, s)
    g = getfield(ggpu, s)
    ratio = g / c
    pct = g / ggpu.total * 100
    if s == :total
        println("-" ^ 70)
    end
    @printf("%-25s %10.1f %10.1f %10.2f %9.1f%%\n", lab, c, g, ratio, pct)
end
println()
println("Pm-3 CPU speedup vs CUFFT(128³): $(round(t_cufft / gcpu.total, digits=2))×")
println("Pm-3 GPU speedup vs CUFFT(128³): $(round(t_cufft / ggpu.total, digits=2))×")
