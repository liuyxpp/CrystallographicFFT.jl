using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, detect_centering_type, CentF, CentI, CentC
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT: plan_centering_fold, centering_fold!, fft_channels!,
    assemble_G0!, ifft_channels!, centering_unfold!, disassemble_G0!
using CrystallographicFFT.KRFFT: SubgridCenteringFoldPlan
using CrystallographicFFT.QFusedKRFFT: M7SCFTPlan, plan_m7_scft, execute_m7_scft!
using CrystallographicFFT.QFusedKRFFT: M2QPlan, plan_m2_q, execute_m2_q!
using CrystallographicFFT.KRFFT: pack_stride_real!
using FFTW
using LinearAlgebra
using Random

"""Generate symmetric field u_sym that respects ops."""
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

@testset "M7 SCFT Round-Trip" begin

    @testset "centering_unfold! roundtrip (fold → unfold recovers f₀)" begin
        for (sg, name, cent) in [
            (70, "Fddd", CentF),
            (229, "Im-3m", CentI),
            (63, "Cmcm", CentC)
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                u_sym = make_symmetric(ops_s, N)

                M = N .÷ 2
                f0 = Float64[u_sym[2i-1,2j-1,2k-1]
                             for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
                f0_backup = copy(f0)

                fold_plan = plan_centering_fold(cent, M)

                # Forward fold
                centering_fold!(fold_plan, f0)

                # Inverse unfold
                centering_unfold!(fold_plan, f0)

                @test maximum(abs.(f0 .- f0_backup)) < 1e-12
            end
        end
    end

    @testset "disassemble_G0! roundtrip (assemble → disassemble)" begin
        for (sg, name, cent) in [
            (70, "Fddd", CentF),
            (229, "Im-3m", CentI)
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                u_sym = make_symmetric(ops_s, N)

                M = N .÷ 2
                f0 = Float64[u_sym[2i-1,2j-1,2k-1]
                             for i in 1:M[1], j in 1:M[2], k in 1:M[3]]

                fold_plan = plan_centering_fold(cent, M)
                centering_fold!(fold_plan, f0)
                fft_channels!(fold_plan)

                # Save channel FFT outputs
                saved = [copy(fold_plan.channel_fft_out[c])
                         for c in 1:fold_plan.n_channels]

                # Assemble
                G0 = zeros(ComplexF64, M)
                assemble_G0!(G0, fold_plan)

                # Disassemble (should recover saved outputs)
                disassemble_G0!(fold_plan, G0)

                for c in 1:fold_plan.n_channels
                    @test fold_plan.channel_fft_out[c] ≈ saved[c] atol=1e-12
                end
            end
        end
    end

    @testset "full FFT roundtrip: fold → FFT → IFFT → unfold" begin
        for (sg, name, cent) in [
            (70, "Fddd", CentF),
            (229, "Im-3m", CentI),
            (63, "Cmcm", CentC)
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                u_sym = make_symmetric(ops_s, N)

                M = N .÷ 2
                f0 = Float64[u_sym[2i-1,2j-1,2k-1]
                             for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
                f0_backup = copy(f0)

                fold_plan = plan_centering_fold(cent, M)

                # fold → FFT → IFFT → unfold should be identity
                centering_fold!(fold_plan, f0)
                fft_channels!(fold_plan)
                ifft_channels!(fold_plan)
                centering_unfold!(fold_plan, f0)

                @test maximum(abs.(f0 .- f0_backup)) < 1e-10
            end
        end
    end

    @testset "execute_m7_scft! matches M2+Q (execute_m2_q!)" begin
        for (sg, name) in [
            (229, "Im-3m"),
            (225, "Fm-3m"),
            (63, "Cmcm"),
            (70, "Fddd")
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                Δs = 0.05
                lattice = Matrix{Float64}(I, 3, 3)

                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                u_sym = make_symmetric(ops_s, N)

                M = N .÷ 2
                f0_m7 = Float64[u_sym[2i-1,2j-1,2k-1]
                                for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
                f0_m2 = copy(f0_m7)

                # --- M2+Q reference ---
                m2q = plan_m2_q(N, sg, 3, Δs, lattice)
                execute_m2_q!(m2q, f0_m2)

                # --- M7 SCFT ---
                m7 = plan_m7_scft(N, sg, 3, Δs, lattice)
                execute_m7_scft!(m7, f0_m7)

                # M7 should match M2+Q to machine precision
                max_diff = maximum(abs.(f0_m7 .- f0_m2))
                @test max_diff < 1e-12
            end
        end
    end
end

