using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using CrystallographicFFT.KRFFT: plan_centering_prefold
using CrystallographicFFT.SpectralIndexing
using FFTW
using LinearAlgebra

# Helper: create symmetric real-space field for a given space group
function make_symmetric(sg_num, N)
    ops = get_ops(sg_num, 3, N)
    u = randn(N...)
    u_sym = zeros(N...)
    for op in ops
        for iz in 0:N[3]-1, iy in 0:N[2]-1, ix in 0:N[1]-1
            x = [ix, iy, iz]
            x_rot = [mod(sum(op.R[d,j]*x[j] for j in 1:3) + op.t[d], N[d]) for d in 1:3]
            u_sym[x_rot[1]+1, x_rot[2]+1, x_rot[3]+1] += u[ix+1, iy+1, iz+1]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

# Helper: end-to-end correctness test
function verify_centered_krfft(sg_num, N; atol=1e-10)
    u_sym = make_symmetric(sg_num, N)
    F_full = fft(u_sym)
    plan = plan_krfft_centered(sg_num, N)
    result = execute_centered_krfft!(plan, u_sym)

    max_err = 0.0
    for (i, pt) in enumerate(plan.full_spec_asu.points)
        h = pt.idx
        F_ref = F_full[h[1]+1, h[2]+1, h[3]+1]
        err = abs(result[i] - F_ref)
        if err > max_err; max_err = err; end
    end
    return max_err, plan
end

@testset "Centering Pre-fold" begin

    @testset "detect_centering_type" begin
        N = (16, 16, 16)
        @test detect_centering_type(get_ops(1, 3, N), N) == CentP      # P1
        @test detect_centering_type(get_ops(221, 3, N), N) == CentP    # Pm-3m
        @test detect_centering_type(get_ops(229, 3, N), N) == CentI    # Im-3m
        @test detect_centering_type(get_ops(225, 3, N), N) == CentF    # Fm-3m
        @test detect_centering_type(get_ops(70, 3, N), N) == CentF     # Fddd
        @test detect_centering_type(get_ops(35, 3, N), N) == CentC     # Cmm2
        @test detect_centering_type(get_ops(139, 3, N), N) == CentI    # I4/mmm
        @test detect_centering_type(get_ops(69, 3, N), N) == CentF     # Fmmm
    end

    @testset "strip_centering" begin
        N = (16, 16, 16)

        # I-centering: 96 → 48 (Oh), all dims halved
        ops_I = get_ops(229, 3, N)
        ops_sub, N_sub = strip_centering(ops_I, CentI, N)
        @test N_sub == (8, 8, 8)
        @test length(ops_sub) == 48

        # F-centering: 192 → 48 (Oh), all dims halved
        ops_F = get_ops(225, 3, N)
        ops_sub_F, N_sub_F = strip_centering(ops_F, CentF, N)
        @test N_sub_F == (8, 8, 8)
        @test length(ops_sub_F) == 48

        # C-centering: halves only x, y
        ops_C = get_ops(35, 3, N)
        ops_sub_C, N_sub_C = strip_centering(ops_C, CentC, N)
        @test N_sub_C == (8, 8, 16)
    end

    @testset "plan_centering_prefold" begin
        N = (16, 16, 16)

        # I-centering: 4 channels
        p = plan_centering_prefold(CentI, N)
        @test p.n_channels == 4
        @test p.N_sub == (8, 8, 8)
        @test length(p.parities) == 4

        # F-centering: 2 channels
        p = plan_centering_prefold(CentF, N)
        @test p.n_channels == 2
        @test p.N_sub == (8, 8, 8)

        # C-centering: 2 channels, partial halving
        p = plan_centering_prefold(CentC, N)
        @test p.n_channels == 2
        @test p.N_sub == (8, 8, 16)

        # A-centering: 2 channels, partial halving
        p = plan_centering_prefold(CentA, N)
        @test p.n_channels == 2
        @test p.N_sub == (16, 8, 8)

        # P-centering: should error
        @test_throws ErrorException plan_centering_prefold(CentP, N)
    end

    @testset "I-centering: SG229 Im-3m" begin
        err, plan = verify_centered_krfft(229, (16, 16, 16))
        @test plan.centering == CentI
        @test plan.n_channels == 4
        @test count(e -> e.channel == 0, plan.merge_table) == 0
        @test err < 1e-10
    end

    @testset "I-centering: SG139 I4/mmm" begin
        err, plan = verify_centered_krfft(139, (16, 16, 16))
        @test plan.centering == CentI
        @test err < 1e-10
    end

    @testset "F-centering: SG225 Fm-3m" begin
        err, plan = verify_centered_krfft(225, (16, 16, 16))
        @test plan.centering == CentF
        @test plan.n_channels == 2
        @test count(e -> e.channel == 0, plan.merge_table) == 0
        @test err < 1e-10
    end

    @testset "F-centering: SG70 Fddd (non-symmorphic)" begin
        err, plan = verify_centered_krfft(70, (16, 16, 16))
        @test plan.centering == CentF
        @test count(e -> e.channel == 0, plan.merge_table) == 0
        @test err < 1e-10
    end

    @testset "F-centering: SG69 Fmmm" begin
        err, plan = verify_centered_krfft(69, (16, 16, 16))
        @test plan.centering == CentF
        @test err < 1e-10
    end

    @testset "C-centering: SG35 Cmm2" begin
        err, plan = verify_centered_krfft(35, (16, 16, 16))
        @test plan.centering == CentC
        @test plan.n_channels == 2
        @test plan.N_sub == (8, 8, 16)
        @test count(e -> e.channel == 0, plan.merge_table) == 0
        @test err < 1e-10
    end

    @testset "Larger grid: SG229 N=32" begin
        err, plan = verify_centered_krfft(229, (32, 32, 32))
        @test err < 1e-10
    end

    @testset "Larger grid: SG225 N=32" begin
        err, plan = verify_centered_krfft(225, (32, 32, 32))
        @test err < 1e-10
    end

end
