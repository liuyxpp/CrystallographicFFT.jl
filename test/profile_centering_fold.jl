# M7 Centering Fold Pipeline — Stage-by-Stage Profiling
#
# Breaks down: pack_stride_real! | centering_fold! | fft_channels! | assemble_G0! | fast_reconstruct!
# Compared against: full FFT baseline (plan_fft + mul!) and plain KRFFT (M2)

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, detect_centering_type
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_krfft_centered, plan_krfft
using CrystallographicFFT.KRFFT: CenteredKRFFTPlan, pack_stride_real!
using CrystallographicFFT.KRFFT: centering_fold!, fft_channels!, assemble_G0!, fast_reconstruct!
using CrystallographicFFT.KRFFT: fft_reconstruct!
using FFTW
using LinearAlgebra: mul!
using Random
using Printf

function make_symmetric(ops, N)
    Random.seed!(42)
    u = randn(N...)
    u_sym = zeros(N...)
    Nv = collect(Int, N)
    for op in ops
        R = round.(Int, op.R); t = round.(Int, op.t)
        for iz in 0:N[3]-1, iy in 0:N[2]-1, ix in 0:N[1]-1
            y = mod.([sum(R[d,:].*[ix,iy,iz])+t[d] for d in 1:3], Nv)
            u_sym[y[1]+1,y[2]+1,y[3]+1] += u[ix+1,iy+1,iz+1]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

function profile_one(sg, name, N_size; nruns=50)
    N = (N_size, N_size, N_size)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, 3, N)
    u = make_symmetric(ops_s, N)
    cent = detect_centering_type(ops_s, N)

    # --- Full FFT baseline ---
    u_c = complex(u)
    fft_out = similar(u_c)
    fft_plan = plan_fft(u_c)
    mul!(fft_out, fft_plan, u_c)  # warmup
    t_fft = minimum([@elapsed(mul!(fft_out, fft_plan, u_c)) for _ in 1:nruns])

    # --- M2 (plain KRFFT) baseline ---
    plan_m2 = plan_krfft(spec, ops_s)
    # Warmup
    plan_m2.input_buffer .= vec(complex(u[1:2:end, 1:2:end, 1:2:end]))
    fft_reconstruct!(plan_m2)
    # Benchmark M2 SCFT fast path
    t_m2 = minimum([@elapsed(fft_reconstruct!(plan_m2)) for _ in 1:nruns])

    # --- M7 (Centered KRFFT) ---
    plan_c = plan_krfft_centered(spec, ops_s)
    if !(plan_c isa CenteredKRFFTPlan)
        @printf "%-10s SG%-3d  %s  NOT APPLICABLE\n" name sg cent
        return
    end

    fold = plan_c.fold_plan
    krfft = plan_c.krfft_plan

    # Warmup all stages
    pack_stride_real!(plan_c.f0_buffer, u)
    centering_fold!(fold, plan_c.f0_buffer)
    fft_channels!(fold)
    G0_view = reshape(krfft.work_buffer, Tuple(krfft.subgrid_dims))
    assemble_G0!(G0_view, fold)
    fast_reconstruct!(krfft)

    # Stage 1: pack_stride_real!
    t_pack = minimum([@elapsed(pack_stride_real!(plan_c.f0_buffer, u)) for _ in 1:nruns])

    # Stage 2: centering_fold!
    pack_stride_real!(plan_c.f0_buffer, u)  # ensure input is valid
    t_fold = minimum([@elapsed(centering_fold!(fold, plan_c.f0_buffer)) for _ in 1:nruns])

    # Stage 3: fft_channels!
    centering_fold!(fold, plan_c.f0_buffer)  # ensure channel bufs valid
    t_fftch = minimum([@elapsed(fft_channels!(fold)) for _ in 1:nruns])

    # Stage 4: assemble_G0!
    fft_channels!(fold)  # ensure fft output valid
    t_assemble = minimum([@elapsed begin
        fill!(G0_view, zero(ComplexF64))
        assemble_G0!(G0_view, fold)
    end for _ in 1:nruns])

    # Actually, assemble_G0! already does fill! inside — let me just time it directly
    t_assemble = minimum([@elapsed(assemble_G0!(G0_view, fold)) for _ in 1:nruns])

    # Stage 5: fast_reconstruct!
    assemble_G0!(G0_view, fold)  # ensure G0 valid
    t_recon = minimum([@elapsed(fast_reconstruct!(krfft)) for _ in 1:nruns])

    # Total M7 (SCFT path: fold + fft + assemble + recon)
    pack_stride_real!(plan_c.f0_buffer, u)
    t_total = t_fold + t_fftch + t_assemble + t_recon
    n_ch = fold.n_channels

    # Also measure fill! alone for M³
    t_fill = minimum([@elapsed(fill!(G0_view, zero(ComplexF64))) for _ in 1:nruns])

    # Print header for each group
    @printf "\n%-10s SG%-3d  |G|=%-3d  %s  %dch  N=%d  M=%d  H=%d\n" name sg length(ops_s) cent n_ch N_size N_size÷2 N_size÷4
    @printf "  Full FFT baseline:    %7.3f ms\n" t_fft*1e3
    @printf "  M2 SCFT (fft+recon):  %7.3f ms  (%5.1f× vs FFT)\n" t_m2*1e3 t_fft/t_m2
    @printf "  ─────────────────────────────────────────────\n"
    @printf "  pack_stride_real!:    %7.3f ms  (%5.1f%%)\n" t_pack*1e3 t_pack/t_total*100
    @printf "  centering_fold!:      %7.3f ms  (%5.1f%%)\n" t_fold*1e3 t_fold/t_total*100
    @printf "  fft_channels!:        %7.3f ms  (%5.1f%%)\n" t_fftch*1e3 t_fftch/t_total*100
    @printf "  assemble_G0!:         %7.3f ms  (%5.1f%%)  [fill!=%.3f ms]\n" t_assemble*1e3 t_assemble/t_total*100 t_fill*1e3
    @printf "  fast_reconstruct!:    %7.3f ms  (%5.1f%%)\n" t_recon*1e3 t_recon/t_total*100
    @printf "  ─────────────────────────────────────────────\n"
    @printf "  M7 total (no pack):   %7.3f ms  (%5.1f× vs FFT)\n" t_total*1e3 t_fft/t_total
    @printf "  M7 理论FFT成本:       %7.3f ms  (n_ch×H³ FFT only)\n" t_fftch*1e3
    @printf "  M7 非FFT开销:         %7.3f ms  (fold+assemble+recon)\n" (t_fold+t_assemble+t_recon)*1e3
    @printf "  M7非FFT/M2非FFT:      fold+asm+recon vs M2 recon-only overhead\n"
end

function main()
    println("=" ^80)
    println("M7 Centering Fold — Stage-by-Stage Profiling")
    println("=" ^80)

    test_cases = [
        (225, "Fm-3m"),    # F, |G|=192
        (229, "Im-3m"),    # I, |G|=96
        (70,  "Fddd"),     # F, |G|=32
        (139, "I4/mmm"),   # I, |G|=32
        (72,  "Ibam"),     # I, |G|=16
        (63,  "Cmcm"),     # C, |G|=16
    ]

    for N_size in [64, 128]
        println("\n" * "=" ^80)
        println("N = $N_size")
        println("=" ^80)
        for (sg, name) in test_cases
            profile_one(sg, name, N_size)
        end
    end
end

main()
