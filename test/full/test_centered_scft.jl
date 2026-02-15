using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_centered_scft, execute_centered_scft!,
    update_kernel!, CenteredSCFTPlan
using CrystallographicFFT.QFusedKRFFT: plan_m7_scft, execute_m7_scft!
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

"""Full-grid FFT diffusion reference: IFFT(K · FFT(u))[subgrid]."""
function fullgrid_reference(u_sym, N, Δs, lattice)
    F_full = fft(u_sym)
    recip_B = 2π * inv(Matrix(lattice))'
    K_full = zeros(ComplexF64, N...)
    for iz in 0:N[3]-1, iy in 0:N[2]-1, ix in 0:N[1]-1
        hc = [ix >= N[1]÷2 ? ix-N[1] : ix,
              iy >= N[2]÷2 ? iy-N[2] : iy,
              iz >= N[3]÷2 ? iz-N[3] : iz]
        kv = recip_B * hc
        K_full[ix+1,iy+1,iz+1] = exp(-dot(kv,kv) * Δs)
    end
    f_out = real.(ifft(K_full .* F_full))
    M = N .÷ 2
    return Float64[f_out[1+(i-1)*2, 1+(j-1)*2, 1+(k-1)*2]
                   for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

"""Extract stride-2 subgrid from full grid."""
function extract_subgrid(u_sym, N)
    M = N .÷ 2
    return Float64[u_sym[1+(i-1)*2, 1+(j-1)*2, 1+(k-1)*2]
                   for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

@testset "CenteredSCFTPlan (fwd+bwd SCFT)" begin

    @testset "fwd+bwd matches full-grid FFT" begin
        for (sg, name) in [
            (225, "Fm-3m"),
            (227, "Fd-3m"),
            (229, "Im-3m"),
            (230, "Ia-3d"),
            (70,  "Fddd"),
            (63,  "Cmcm"),
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                Δs = 0.05
                lattice = Matrix{Float64}(I, 3, 3)

                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                u_sym = make_symmetric(ops_s, N)

                f0 = extract_subgrid(u_sym, N)
                f0_ref = fullgrid_reference(u_sym, N, Δs, lattice)

                # fwd+bwd
                spec_asu = calc_spectral_asu(ops_s, 3, N)
                scft = plan_centered_scft(spec_asu, ops_s, N, Δs, lattice)
                execute_centered_scft!(scft, f0)

                max_diff = maximum(abs.(f0 .- f0_ref))
                @test max_diff < 1e-12
            end
        end
    end

    @testset "fwd+bwd matches M7+Q" begin
        for (sg, name) in [
            (229, "Im-3m"),
            (225, "Fm-3m"),
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                Δs = 0.05
                lattice = Matrix{Float64}(I, 3, 3)

                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                u_sym = make_symmetric(ops_s, N)

                f0_fwd = extract_subgrid(u_sym, N)
                f0_m7  = copy(f0_fwd)

                m7 = plan_m7_scft(N, sg, 3, Δs, lattice)
                execute_m7_scft!(m7, f0_m7)

                spec_asu = calc_spectral_asu(ops_s, 3, N)
                scft = plan_centered_scft(spec_asu, ops_s, N, Δs, lattice)
                execute_centered_scft!(scft, f0_fwd)

                # M7+Q may have Q-matrix conditioning error — compare both
                # against full FFT first, then check they're close
                f0_ref = fullgrid_reference(
                    make_symmetric(ops_s, N), N, Δs, lattice)
                err_fwd = maximum(abs.(f0_fwd .- f0_ref))
                err_m7  = maximum(abs.(f0_m7 .- f0_ref))

                @test err_fwd < 1e-12
                @test err_m7 < 1e-10  # M7+Q may have slightly larger error
            end
        end
    end

    @testset "K=1 roundtrip (no diffusion)" begin
        for (sg, name) in [
            (229, "Im-3m"),
            (225, "Fm-3m"),
            (227, "Fd-3m"),
            (70,  "Fddd"),
            (63,  "Cmcm"),
        ]
            @testset "$name (SG$sg)" begin
                N = (32, 32, 32)
                lattice = Matrix{Float64}(I, 3, 3)

                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                u_sym = make_symmetric(ops_s, N)

                f0 = extract_subgrid(u_sym, N)
                f0_orig = copy(f0)

                # Δs = 0 means K(h) = 1 for all h
                spec_asu = calc_spectral_asu(ops_s, 3, N)
                scft = plan_centered_scft(spec_asu, ops_s, N, 0.0, lattice)
                execute_centered_scft!(scft, f0)

                @test maximum(abs.(f0 .- f0_orig)) < 1e-12
            end
        end
    end

    @testset "update_kernel!" begin
        N = (32, 32, 32)
        sg = 229
        lattice = Matrix{Float64}(I, 3, 3)

        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        u_sym = make_symmetric(ops_s, N)

        spec_asu = calc_spectral_asu(ops_s, 3, N)

        # Build with Δs=0.05, then update to Δs=0.10
        scft = plan_centered_scft(spec_asu, ops_s, N, 0.05, lattice)
        f0_ds05 = extract_subgrid(u_sym, N)
        execute_centered_scft!(scft, f0_ds05)

        update_kernel!(scft, spec_asu, N, 0.10, lattice)
        f0_ds10 = extract_subgrid(u_sym, N)
        execute_centered_scft!(scft, f0_ds10)

        # Fresh plan with Δs=0.10
        scft2 = plan_centered_scft(spec_asu, ops_s, N, 0.10, lattice)
        f0_ref = extract_subgrid(u_sym, N)
        execute_centered_scft!(scft2, f0_ref)

        @test maximum(abs.(f0_ds10 .- f0_ref)) < 1e-14
        @test maximum(abs.(f0_ds05 .- f0_ds10)) > 1e-4  # Different Δs ⟹ different result
    end
end
