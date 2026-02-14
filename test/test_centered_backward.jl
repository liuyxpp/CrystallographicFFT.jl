using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, detect_centering_type
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT
using FFTW
using Random

const CKRFFT = CrystallographicFFT.KRFFT

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

@testset "Centered KRFFT Backward" begin

    @testset "Roundtrip: fft_reconstruct_centered! → ifft_unrecon_centered!" begin
        for (sg, name) in [
            (225, "Fm-3m"),
            (227, "Fd-3m"),
            (229, "Im-3m"),
            (230, "Ia-3d"),
            (139, "I4/mmm"),
            (63,  "Cmcm")
        ]
            @testset "$name (SG $sg)" begin
                N = (16, 16, 16)
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                spec = calc_spectral_asu(ops_s, 3, N)

                u_sym = make_symmetric(ops_s, N)

                fwd = CKRFFT.plan_krfft_centered(spec, ops_s)
                @test fwd isa CKRFFT.CenteredKRFFTPlan

                bwd = CKRFFT.plan_centered_ikrfft(spec, ops_s, fwd)
                @test bwd isa CKRFFT.CenteredKRFFTBackwardPlan

                # Extract f₀ subgrid
                M = N .÷ 2
                f0 = Float64[u_sym[2i-1,2j-1,2k-1]
                             for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
                f0_backup = copy(f0)

                # Forward: f₀ → F̂
                CKRFFT.pack_stride_real!(fwd.f0_buffer, u_sym)
                F_spec = copy(CKRFFT.fft_reconstruct_centered!(fwd))

                # Backward: F̂ → f₀'
                f0_out = zeros(Float64, M...)
                CKRFFT.execute_centered_ikrfft!(bwd, F_spec, f0_out)

                max_err = maximum(abs.(f0_out .- f0_backup))
                @test max_err < 1e-10
            end
        end
    end

    @testset "Spectral consistency: forward(backward(F̂)) ≈ F̂" begin
        for (sg, name) in [
            (225, "Fm-3m"),
            (229, "Im-3m"),
            (63,  "Cmcm"),
            (70,  "Fddd")
        ]
            @testset "$name (SG $sg)" begin
                N = (16, 16, 16)
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                spec = calc_spectral_asu(ops_s, 3, N)

                u_sym = make_symmetric(ops_s, N)

                fwd = CKRFFT.plan_krfft_centered(spec, ops_s)
                bwd = CKRFFT.plan_centered_ikrfft(spec, ops_s, fwd)

                # Forward: f₀ → F̂
                CKRFFT.pack_stride_real!(fwd.f0_buffer, u_sym)
                F_orig = copy(CKRFFT.fft_reconstruct_centered!(fwd))

                # Backward: F̂ → f₀'
                CKRFFT.ifft_unrecon_centered!(bwd, F_orig)

                # Forward again: f₀' → F̂'
                F_roundtrip = copy(CKRFFT.fft_reconstruct_centered!(fwd))

                max_err = maximum(abs.(F_roundtrip .- F_orig))
                @test max_err < 1e-10
            end
        end
    end

    @testset "ifft_unrecon_centered! writes to f0_buffer" begin
        N = (16, 16, 16)
        ops = get_ops(229, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        spec = calc_spectral_asu(ops_s, 3, N)
        u_sym = make_symmetric(ops_s, N)

        fwd = CKRFFT.plan_krfft_centered(spec, ops_s)
        bwd = CKRFFT.plan_centered_ikrfft(spec, ops_s, fwd)

        CKRFFT.pack_stride_real!(fwd.f0_buffer, u_sym)
        F_spec = copy(CKRFFT.fft_reconstruct_centered!(fwd))

        # Use convenience wrapper
        CKRFFT.ifft_unrecon_centered!(bwd, F_spec)

        # Result should be in bwd.f0_buffer (which is fwd.f0_buffer)
        M = N .÷ 2
        f0_ref = Float64[u_sym[2i-1,2j-1,2k-1]
                         for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
        max_err = maximum(abs.(bwd.f0_buffer .- f0_ref))
        @test max_err < 1e-10
    end
end
