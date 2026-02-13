using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, detect_centering_type, CentF, CentI, CentC
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT: plan_krfft_centered, execute_centered_krfft!
using CrystallographicFFT.KRFFT: plan_centering_fold, centering_fold!, fft_channels!, assemble_G0!
using CrystallographicFFT.KRFFT: SubgridCenteringFoldPlan, CenteredKRFFTPlan
using CrystallographicFFT.KRFFT: pack_stride_real!
using FFTW
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

@testset "Centering Fold on Stride-2 Subgrid" begin

    @testset "plan_centering_fold creates correct plan" begin
        @testset "F-centering (2 channels)" begin
            plan = plan_centering_fold(CentF, (32,32,32))
            @test plan.centering == CentF
            @test plan.n_channels == 2
            @test plan.H == (16,16,16)
            @test plan.offsets == [(0,0,0), (1,1,1)]
        end

        @testset "I-centering (4 channels)" begin
            plan = plan_centering_fold(CentI, (32,32,32))
            @test plan.centering == CentI
            @test plan.n_channels == 4
            @test plan.H == (16,16,16)
            @test plan.offsets == [(0,0,0), (1,1,0), (1,0,1), (0,1,1)]
        end

        @testset "C-centering (4 channels)" begin
            plan = plan_centering_fold(CentC, (32,32,32))
            @test plan.centering == CentC
            @test plan.n_channels == 4
            @test plan.H == (16,16,16)
        end
    end

    @testset "centering_fold! + assemble_G0! vs direct FFT" begin
        # Test: fold + channel FFT + assemble should equal direct FFT of f₀
        for (sg, name, cent_expected) in [
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
                f0 = Float64[u_sym[2i-1,2j-1,2k-1] for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
                G0_ref = fft(f0)

                fold_plan = plan_centering_fold(cent_expected, M)
                centering_fold!(fold_plan, f0)
                fft_channels!(fold_plan)

                G0_assembled = zeros(ComplexF64, M)
                assemble_G0!(G0_assembled, fold_plan)

                # Check alive entries match, extinct entries are zero
                for iz in 0:M[3]-1, iy in 0:M[2]-1, ix in 0:M[1]-1
                    @test G0_assembled[ix+1,iy+1,iz+1] ≈ G0_ref[ix+1,iy+1,iz+1] atol=1e-10
                end
            end
        end
    end

    @testset "execute_centered_krfft! vs full FFT" begin
        for (sg, name) in [
            (70, "Fddd"),
            (229, "Im-3m"),
            (225, "Fm-3m"),
            (139, "I4/mmm"),
            (63, "Cmcm"),
            (72, "Ibam")
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                spec = calc_spectral_asu(ops_s, 3, N)
                u_sym = make_symmetric(ops_s, N)

                F_ref = fft(u_sym)

                plan = plan_krfft_centered(spec, ops_s)
                @test plan isa CenteredKRFFTPlan

                execute_centered_krfft!(plan, u_sym)
                spec_out = plan.krfft_plan.output_buffer

                max_err = 0.0
                for (i, _) in enumerate(spec.points)
                    h = get_k_vector(spec, i)
                    fref = F_ref[mod(h[1],N[1])+1, mod(h[2],N[2])+1, mod(h[3],N[3])+1]
                    max_err = max(max_err, abs(spec_out[i] - fref))
                end
                @test max_err < 1e-8
            end
        end
    end

    @testset "pack_stride_real! correctness" begin
        N = (16, 16, 16)
        M = N .÷ 2
        u = randn(N...)
        f0 = zeros(Float64, M)
        pack_stride_real!(f0, u)
        for k in 1:M[3], j in 1:M[2], i in 1:M[1]
            @test f0[i,j,k] == u[2i-1, 2j-1, 2k-1]
        end
    end

    @testset "fft_reconstruct_centered! matches execute" begin
        N = (32, 32, 32)
        ops = get_ops(229, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u_sym = make_symmetric(ops_s, N)

        plan = plan_krfft_centered(spec, ops_s)
        @test plan isa CenteredKRFFTPlan

        # Execute full pipeline
        execute_centered_krfft!(plan, u_sym)
        result_exec = copy(plan.krfft_plan.output_buffer)

        # Now use fft_reconstruct_centered! manually
        using CrystallographicFFT.KRFFT: fft_reconstruct_centered!
        pack_stride_real!(plan.f0_buffer, u_sym)
        fft_reconstruct_centered!(plan)
        result_recon = copy(plan.krfft_plan.output_buffer)

        @test result_exec ≈ result_recon atol=1e-12
    end
end
