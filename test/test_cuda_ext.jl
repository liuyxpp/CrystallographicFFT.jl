# GPU CUDA Extension Tests for CrystallographicFFT
#
# Prerequisites: CUDA.jl must be installed and a GPU must be available.
# Run: julia --project=test test/test_cuda_ext.jl
#
# NOT included in runtests.jl (requires GPU hardware).

using Test
using CUDA
using FFTW
using LinearAlgebra
using CrystallographicFFT

# Skip if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

@info "GPU detected: $(CUDA.name(CUDA.device()))"

# ── Test groups ──────────────────────────────────────────────────────────────

general_groups = [
    (225, "Pm-3m"),
    (200, "Pm-3"),
]

centered_groups = [
    (225, "Fm-3m"),
    (229, "Im-3m"),
]

N_test = (16, 16, 16)

@testset "GPU CuArray Extension" begin

    # ── 1. Roundtrip: General Path (GPU matches CPU) ────────────────────────
    @testset "GPU General: $name (SG $sg)" for (sg, name) in general_groups
        plan_gpu = plan_cfft_pair(N_test, sg, 3;
                                   method=:general,
                                   array_type=CuArray{Float64})
        plan_cpu = plan_cfft_pair(N_test, sg, 3; method=:general)

        M = plan_cpu.M
        n_spec = plan_cpu.n_spec
        f0_cpu = randn(M...)

        # CPU reference
        F_cpu = zeros(ComplexF64, n_spec)
        f0_back_cpu = zeros(Float64, M...)
        cfft!(F_cpu, plan_cpu, f0_cpu)
        icfft!(f0_back_cpu, plan_cpu, F_cpu)

        # GPU path
        f0_gpu = CuArray(f0_cpu)
        F_gpu = CUDA.zeros(ComplexF64, n_spec)
        f0_back_gpu = CUDA.zeros(Float64, M...)

        cfft!(F_gpu, plan_gpu, f0_gpu)
        icfft!(f0_back_gpu, plan_gpu, F_gpu)

        # GPU vs CPU comparison (not vs original, since non-symmetric input)
        F_host = Array(F_gpu)
        f0_host = Array(f0_back_gpu)

        fwd_diff = maximum(abs.(F_host .- F_cpu))
        bwd_diff = maximum(abs.(f0_host .- f0_back_cpu))

        @info "  General $name: fwd_diff=$fwd_diff, bwd_diff=$bwd_diff"
        @test fwd_diff < 1e-8
        @test bwd_diff < 1e-8
    end

    # ── 2. Roundtrip: Centered Path (GPU matches CPU) ───────────────────────
    @testset "GPU Centered: $name (SG $sg)" for (sg, name) in centered_groups
        plan_gpu = plan_cfft_pair(N_test, sg, 3;
                                   method=:centered,
                                   array_type=CuArray{Float64})
        plan_cpu = plan_cfft_pair(N_test, sg, 3; method=:centered)

        M = plan_cpu.M
        n_spec = plan_cpu.n_spec
        f0_cpu = randn(M...)

        # CPU reference
        F_cpu = zeros(ComplexF64, n_spec)
        f0_back_cpu = zeros(Float64, M...)
        cfft!(F_cpu, plan_cpu, f0_cpu)
        icfft!(f0_back_cpu, plan_cpu, F_cpu)

        # GPU path
        f0_gpu = CuArray(f0_cpu)
        F_gpu = CUDA.zeros(ComplexF64, n_spec)
        f0_back_gpu = CUDA.zeros(Float64, M...)

        cfft!(F_gpu, plan_gpu, f0_gpu)
        icfft!(f0_back_gpu, plan_gpu, F_gpu)

        # Compare
        F_host = Array(F_gpu)
        f0_host = Array(f0_back_gpu)

        fwd_diff = maximum(abs.(F_host .- F_cpu))
        bwd_diff = maximum(abs.(f0_host .- f0_back_cpu))

        @info "  Centered $name: fwd_diff=$fwd_diff, bwd_diff=$bwd_diff"
        @test fwd_diff < 1e-8
        @test bwd_diff < 1e-8
    end

    # ── 3. Spectral Consistency (forward-only) ──────────────────────────────
    @testset "GPU Spectral: $name (SG $sg)" for (sg, name) in
        [(225, "Pm-3m"), (229, "Im-3m")]
        plan_gpu = plan_cfft(N_test, sg, 3;
                              method=:general,
                              array_type=CuArray{Float64})
        plan_cpu = plan_cfft(N_test, sg, 3; method=:general)

        M = plan_cpu.M
        n_spec = plan_cpu.n_spec
        f0_cpu = randn(M...)
        f0_gpu = CuArray(f0_cpu)

        F_cpu = zeros(ComplexF64, n_spec)
        F_gpu = CUDA.zeros(ComplexF64, n_spec)

        cfft!(F_cpu, plan_cpu, f0_cpu)
        cfft!(F_gpu, plan_gpu, f0_gpu)

        F_host = Array(F_gpu)
        max_diff = maximum(abs.(F_host .- F_cpu))
        @info "  Spectral $name: max_diff=$max_diff"
        @test max_diff < 1e-8
    end

    # ── 4. Forward-only plan ────────────────────────────────────────────────
    @testset "GPU Forward-only plan" begin
        plan = plan_cfft(N_test, 225, 3;
                          method=:general,
                          array_type=CuArray{Float64})
        M = plan.M
        f0 = CuArray(randn(M...))
        F = CUDA.zeros(ComplexF64, plan.n_spec)
        cfft!(F, plan, f0)
        @test all(isfinite, Array(F))
    end

    # ── 5. Backward-only plan ───────────────────────────────────────────────
    @testset "GPU Backward-only plan" begin
        plan_f = plan_cfft(N_test, 225, 3;
                            method=:general,
                            array_type=CuArray{Float64})
        plan_b = plan_icfft(plan_f)
        M = plan_f.M

        plan_f_cpu = plan_cfft(N_test, 225, 3; method=:general)
        plan_b_cpu = plan_icfft(plan_f_cpu)

        f0_cpu = randn(M...)
        f0 = CuArray(f0_cpu)

        # CPU reference
        F_cpu = zeros(ComplexF64, plan_f.n_spec)
        cfft!(F_cpu, plan_f_cpu, f0_cpu)
        f0_back_cpu = zeros(Float64, M...)
        icfft!(f0_back_cpu, plan_b_cpu, F_cpu)

        # GPU
        F = CUDA.zeros(ComplexF64, plan_f.n_spec)
        cfft!(F, plan_f, f0)
        f0_back = CUDA.zeros(Float64, M...)
        icfft!(f0_back, plan_b, F)

        diff = maximum(abs.(Array(f0_back) .- f0_back_cpu))
        @info "  Backward-only diff=$diff"
        @test diff < 1e-8
    end
end
