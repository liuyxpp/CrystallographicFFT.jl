# Proof-of-concept: M7 forward + K·ASU + M7 backward SCFT
# Compare M7+Q vs recon+K+inv_recon decomposition
# Usage: julia --project=test test/debug_m7_fkb.jl

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.QFusedKRFFT
using CrystallographicFFT.KRFFT: plan_centering_fold, centering_fold!, fft_channels!,
    assemble_G0!, ifft_channels!, centering_unfold!, disassemble_G0!,
    GeneralCFFTPlan, plan_krfft, ReconEntry, fast_reconstruct!
using FFTW
using Statistics
using LinearAlgebra
using Printf
using Random

const SI = CrystallographicFFT.SpectralIndexing

function bench(f; n_warmup=5, n_trials=20)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000
end

function run_poc(sg, name, N_size)
    N = (N_size, N_size, N_size)
    dim = 3; Δs = 0.05
    lattice = Matrix{Float64}(I, 3, 3)

    println("\n" * "="^80)
    println("PoC: $name (SG $sg), N=$N")
    println("="^80)

    ops = get_ops(sg, dim, N)
    _, ops_s = find_optimal_shift(ops, N)

    # --- Plans ---
    m7q = plan_m7_scft(N, sg, dim, Δs, lattice)
    M = Tuple(m7q.M)
    fold = m7q.fold_plan
    m2q = m7q.m2q_plan

    # Random f₀ on M-grid (no symmetrization needed for timing/correctness of path comparison)
    Random.seed!(42)
    f0_orig = rand(M...)

    # Build plan_krfft for M-grid spectral ASU
    spec = SI.calc_spectral_asu(ops_s, dim, M)
    plan_gen = plan_krfft(spec, ops_s)
    n_spec = length(spec.points)
    n_ops = plan_gen.n_ops
    M_sub = plan_gen.subgrid_dims
    M_vol = prod(M)

    println("M=$M, M_sub=$M_sub, n_spec=$n_spec, n_ops=$n_ops")
    println("M³=$(M_vol), ASU ratio=$(round(M_vol/n_spec, digits=1))")

    # K(h) for spectral ASU
    recip_B = 2π * inv(lattice)'
    K_asu = zeros(n_spec)
    for h_idx in 1:n_spec
        hv = SI.get_k_vector(spec, h_idx)
        hf = Float64[hv[d] >= M[d]÷2 ? hv[d]-M[d] : hv[d] for d in 1:dim]
        K_asu[h_idx] = exp(-dot(recip_B * hf, recip_B * hf) * Δs)
    end

    # Build CSR inverse recon table
    println("Building inverse recon table...")
    table = plan_gen.recon_table
    counts = zeros(Int, M_vol)
    for h in 1:n_spec, g in 1:n_ops
        counts[table[g,h].buffer_idx] += 1
    end
    row_ptr = zeros(Int, M_vol+1)
    row_ptr[1] = 1
    for q in 1:M_vol; row_ptr[q+1] = row_ptr[q] + counts[q]; end
    total = row_ptr[end]-1

    inv_hidx = zeros(Int32, total)
    inv_wt = zeros(ComplexF64, total)
    fill!(counts, 0)
    for h in 1:n_spec, g in 1:n_ops
        e = table[g,h]
        q = e.buffer_idx
        p = row_ptr[q] + counts[q]
        inv_hidx[p] = Int32(h)
        inv_wt[p] = conj(e.weight) / n_ops
        counts[q] += 1
    end
    avg_e = round(total/M_vol, digits=1)
    max_e = maximum(row_ptr[q+1]-row_ptr[q] for q in 1:M_vol)
    println("Inverse table: avg=$avg_e entries/q, max=$max_e, total=$total")

    # === Method B: fwd+K+bwd ===
    function method_b!(f0_out, f0_in)
        copyto!(f0_out, f0_in)
        centering_fold!(fold, f0_out)
        fft_channels!(fold)
        assemble_G0!(reshape(plan_gen.work_buffer, Tuple(M_sub)), fold)

        F_spec = fast_reconstruct!(plan_gen)
        @inbounds for i in 1:n_spec; F_spec[i] *= K_asu[i]; end

        G_out = plan_gen.work_buffer
        @inbounds for q in 1:M_vol
            val = zero(ComplexF64)
            for p in row_ptr[q]:(row_ptr[q+1]-1)
                val += inv_wt[p] * F_spec[inv_hidx[p]]
            end
            G_out[q] = val
        end

        disassemble_G0!(fold, reshape(G_out, Tuple(M_sub)))
        ifft_channels!(fold)
        centering_unfold!(fold, f0_out)
    end

    # --- Correctness ---
    f0_a = copy(f0_orig); execute_m7_scft!(m7q, f0_a)
    f0_b = copy(f0_orig); method_b!(f0_b, f0_orig)
    err = norm(f0_b - f0_a) / norm(f0_a)
    println("\nError vs M7+Q: $err  $(err < 1e-8 ? "✓ PASS" : "✗ FAIL")")

    # --- Timing ---
    println("\n--- Timing (median of 20 trials) ---")
    f0_t = copy(f0_orig)

    t_a = bench() do; copyto!(f0_t, f0_orig); execute_m7_scft!(m7q, f0_t); end
    t_b = bench() do; method_b!(f0_t, f0_orig); end

    G0v = reshape(plan_gen.work_buffer, Tuple(M_sub))
    t_fold = bench() do
        copyto!(f0_t, f0_orig); centering_fold!(fold, f0_t)
        fft_channels!(fold); assemble_G0!(G0v, fold)
    end
    t_fwd = bench() do; fast_reconstruct!(plan_gen); end

    Fr = copy(plan_gen.output_buffer); Gw = plan_gen.work_buffer
    t_inv = bench() do
        @inbounds for q in 1:M_vol
            val = zero(ComplexF64)
            for p in row_ptr[q]:(row_ptr[q+1]-1)
                val += inv_wt[p] * Fr[inv_hidx[p]]
            end
            Gw[q] = val
        end
    end

    t_back = bench() do
        disassemble_G0!(fold, G0v); ifft_channels!(fold)
        centering_unfold!(fold, f0_t)
    end

    G0q = reshape(m2q.Y_buf, M)
    t_qmul = bench() do
        assemble_G0!(G0q, fold)
        if m2q.is_separable
            CrystallographicFFT.QFusedKRFFT._q_multiply_separable!(
                m2q.Y_new_buf, m2q.Y_buf, m2q)
        else
            CrystallographicFFT.QFusedKRFFT._q_multiply_generic!(
                m2q.Y_new_buf, m2q.Y_buf, m2q)
        end
    end

    @printf("  %-25s %8.3f ms\n", "M7+Q total", t_a)
    @printf("  %-25s %8.3f ms\n", "  Q-multiply", t_qmul)
    @printf("  %-25s %8.3f ms\n", "fwd+K+bwd total", t_b)
    @printf("  %-25s %8.3f ms\n", "  fold+FFT+assemble", t_fold)
    @printf("  %-25s %8.3f ms\n", "  forward recon", t_fwd)
    @printf("  %-25s %8.3f ms\n", "  inverse recon (naive)", t_inv)
    @printf("  %-25s %8.3f ms\n", "  disasm+IFFT+unfold", t_back)
    println()
    @printf("  inv/fwd recon ratio:     %.2f×\n", t_inv / t_fwd)
    @printf("  naive inv / Q-mul ratio: %.2f×\n", t_inv / t_qmul)
    @printf("  Speedup M7+Q / fwd+K+bwd: %.2f×\n", t_a / t_b)
end

FFTW.set_num_threads(1)
run_poc(225, "Fm-3m", 64)
run_poc(225, "Fm-3m", 128)
