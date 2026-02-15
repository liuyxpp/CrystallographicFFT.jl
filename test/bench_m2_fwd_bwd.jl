"""
M2 Forward vs M2 Backward benchmark.

Compares:
  - M2 Forward:  pack → FFT → reconstruct  (execute_krfft!)
  - M2 Backward: inv_reconstruct → IFFT     (execute_m2_backward!)

Also measures the sub-steps separately:
  - Forward: FFT + reconstruct (fft_reconstruct! on pre-packed buffer)
  - Backward: inv_reconstruct + IFFT
"""

using CrystallographicFFT
using CrystallographicFFT.KRFFT: GeneralCFFTPlan, plan_krfft, fft_reconstruct!,
    plan_m2_backward, execute_m2_backward!, _inv_reconstruct_m2!, M2BackwardPlan,
    pack_stride!, auto_L
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using FFTW
using LinearAlgebra
using Printf
using Statistics
using Random

FFTW.set_num_threads(1)

function bench_fwd_bwd(sg_num, name, N_tuple; n_trials=30, warmup=5)
    dim = 3
    ops = get_ops(sg_num, dim, N_tuple)
    _, ops_s = find_optimal_shift(ops, N_tuple)
    spec = calc_spectral_asu(ops_s, dim, N_tuple)
    n_spec = length(spec.points)

    # Forward plan
    fwd = plan_krfft(spec, ops_s)
    L = auto_L(ops_s)
    M = [N_tuple[d] ÷ L[d] for d in 1:dim]
    M_vol = prod(M)

    # Backward plan
    bwd = plan_m2_backward(spec, ops_s)

    # Input data
    u = rand(Float64, N_tuple...)

    # Pre-pack for forward (avoid measuring pack cost)
    pack_stride!(fwd, u)

    # Build F_spec for backward
    fft_reconstruct!(fwd)
    F_spec = copy(fwd.output_buffer[1:n_spec])

    # === Sub-step timing ===
    sub_in = reshape(fwd.input_buffer, Tuple(fwd.subgrid_dims))
    sub_out = reshape(fwd.work_buffer, Tuple(fwd.subgrid_dims))

    t_fwd_fft = zeros(n_trials)
    t_fwd_recon = zeros(n_trials)
    t_bwd_irecon = zeros(n_trials)
    t_bwd_ifft = zeros(n_trials)
    t_fwd_total = zeros(n_trials)
    t_bwd_total = zeros(n_trials)

    Y_in = reshape(bwd.Y_buf, Tuple(bwd.subgrid_dims))
    f_out = reshape(bwd.f0_buf, Tuple(bwd.subgrid_dims))

    for trial in 1:(warmup + n_trials)
        # Re-pack
        pack_stride!(fwd, u)

        # Forward: FFT
        t1 = time_ns()
        mul!(sub_out, fwd.sub_plans[1], sub_in)
        t2 = time_ns()

        # Forward: reconstruct
        CrystallographicFFT.KRFFT.fast_reconstruct!(fwd)
        t3 = time_ns()

        # Backward: inv_reconstruct
        rand!(F_spec)  # different input each time
        t4 = time_ns()
        _inv_reconstruct_m2!(bwd, F_spec)
        t5 = time_ns()

        # Backward: IFFT
        mul!(f_out, bwd.ifft_plan, Y_in)
        t6 = time_ns()

        if trial > warmup
            i = trial - warmup
            t_fwd_fft[i] = (t2 - t1) / 1e3
            t_fwd_recon[i] = (t3 - t2) / 1e3
            t_bwd_irecon[i] = (t5 - t4) / 1e3
            t_bwd_ifft[i] = (t6 - t5) / 1e3
            t_fwd_total[i] = (t3 - t1) / 1e3
            t_bwd_total[i] = (t6 - t4) / 1e3
        end
    end

    return Dict(
        "fwd_fft" => median(t_fwd_fft),
        "fwd_recon" => median(t_fwd_recon),
        "bwd_irecon" => median(t_bwd_irecon),
        "bwd_ifft" => median(t_bwd_ifft),
        "fwd_total" => median(t_fwd_total),
        "bwd_total" => median(t_bwd_total),
    ), n_spec, M_vol, bwd.d
end

groups = [
    (225, "Fm-3m"),
    (221, "Pm-3m"),
    (200, "Pm-3"),
    (227, "Fd-3m"),
    (229, "Im-3m"),
    (230, "Ia-3d"),
    (70,  "Fddd"),
    (63,  "Cmcm"),
    (72,  "Ibam"),
    (74,  "Imma"),
    (139, "I4/mmm"),
    (123, "P4/mmm"),
    (47,  "Pmmm"),
    (10,  "P2/m"),
]

for N_val in [64, 128]
    N_tuple = (N_val, N_val, N_val)
    n_trials = N_val <= 64 ? 30 : 10

    println("=" ^ 120)
    @printf("M2 Forward vs Backward Benchmark (N=%d, %d trials, single-threaded FFTW)\n", N_val, n_trials)
    println("=" ^ 120)
    @printf("%-8s  %4s  %6s  %5s  %8s  %8s  %8s | %8s  %8s  %8s | %6s\n",
            "Group", "SG", "n_spec", "d",
            "fwd_FFT", "fwd_rec", "fwd_tot",
            "bwd_irec", "bwd_IFFT", "bwd_tot",
            "bwd/fwd")
    println("-" ^ 120)

    for (sg, name) in groups
        t, n_spec, M_vol, d = bench_fwd_bwd(sg, name, N_tuple; n_trials=n_trials)
        @printf("%-8s  %4d  %6d  %5d  %7.0fμs  %7.0fμs  %7.0fμs | %7.0fμs  %7.0fμs  %7.0fμs | %5.2f×\n",
                name, sg, n_spec, d,
                t["fwd_fft"], t["fwd_recon"], t["fwd_total"],
                t["bwd_irecon"], t["bwd_ifft"], t["bwd_total"],
                t["bwd_total"] / t["fwd_total"])
    end
    println()
end
