using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.QFusedKRFFT: M2SCFTPlan, plan_m2_scft, execute_m2_scft!,
    update_m2_kernel!
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_centered_scft, execute_centered_scft!, auto_L
using FFTW
using LinearAlgebra
using Random

"""
Generate symmetric field respecting ops. Allocation-free inner loop.
"""
function make_symmetric(ops, N)
    Random.seed!(42)
    u = randn(N...)
    u_sym = zeros(N...)
    N1, N2, N3 = N
    y = zeros(Int, 3)
    for op in ops
        R = round.(Int, op.R)
        t = round.(Int, op.t)
        @inbounds for iz in 0:N3-1, iy in 0:N2-1, ix in 0:N1-1
            for d in 1:3
                y[d] = mod(R[d,1]*ix + R[d,2]*iy + R[d,3]*iz + t[d], N[d])
            end
            u_sym[y[1]+1, y[2]+1, y[3]+1] += u[ix+1, iy+1, iz+1]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

"""Full-grid FFT diffusion reference. Returns stride-L subgrid."""
function fullgrid_reference(u_sym, N, Δs, lattice, L)
    F_full = fft(u_sym)
    recip_B = 2π * inv(Matrix(lattice))'
    K_full = zeros(ComplexF64, N...)
    N1, N2, N3 = N
    hc = zeros(3)
    @inbounds for iz in 0:N3-1, iy in 0:N2-1, ix in 0:N1-1
        hc[1] = ix >= N1÷2 ? ix - N1 : ix
        hc[2] = iy >= N2÷2 ? iy - N2 : iy
        hc[3] = iz >= N3÷2 ? iz - N3 : iz
        kv = recip_B * hc
        K_full[ix+1,iy+1,iz+1] = exp(-dot(kv,kv) * Δs)
    end
    f_out = real.(ifft(K_full .* F_full))
    M = N .÷ Tuple(L)
    return Float64[f_out[1+(i-1)*L[1], 1+(j-1)*L[2], 1+(k-1)*L[3]]
                   for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

"""Extract stride-L subgrid (L may be anisotropic)."""
function extract_subgrid(u_sym, N, L)
    M = N .÷ Tuple(L)
    return Float64[u_sym[1+(i-1)*L[1], 1+(j-1)*L[2], 1+(k-1)*L[3]]
                   for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

@testset "M2SCFTPlan (fwd+bwd SCFT, all groups)" begin

    N = (16, 16, 16)
    Δs = 0.05
    lattice = Matrix{Float64}(I, 3, 3)

    @testset "M2 fwd+bwd vs Full FFT" begin
        for (sg, name) in [
            (221, "Pm-3m"),
            (123, "P4/mmm"),
            (47,  "Pmmm"),
            (10,  "P2/m"),
            (225, "Fm-3m"),
            (229, "Im-3m"),
            (227, "Fd-3m"),
            (70,  "Fddd"),
            (63,  "Cmcm"),
        ]
            @testset "$name (SG$sg)" begin
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                L = auto_L(ops_s)
                u_sym = make_symmetric(ops_s, N)

                f0 = extract_subgrid(u_sym, N, L)
                f0_ref = fullgrid_reference(u_sym, N, Δs, lattice, L)

                scft = plan_m2_scft(N, sg, 3, Δs, lattice)
                execute_m2_scft!(scft, f0)

                max_diff = maximum(abs.(f0 .- f0_ref))
                @test max_diff < 1e-10
            end
        end
    end

    @testset "K=1 roundtrip" begin
        for (sg, name) in [(221, "Pm-3m"), (225, "Fm-3m"), (227, "Fd-3m")]
            @testset "$name" begin
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                L = auto_L(ops_s)
                u_sym = make_symmetric(ops_s, N)

                f0 = extract_subgrid(u_sym, N, L)
                f0_orig = copy(f0)

                scft = plan_m2_scft(N, sg, 3, 0.0, lattice)
                execute_m2_scft!(scft, f0)

                @test maximum(abs.(f0 .- f0_orig)) < 1e-12
            end
        end
    end

    @testset "update_m2_kernel!" begin
        sg = 221
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        L = auto_L(ops_s)
        u_sym = make_symmetric(ops_s, N)

        scft = plan_m2_scft(N, sg, 3, 0.05, lattice)
        f0_ds05 = extract_subgrid(u_sym, N, L)
        execute_m2_scft!(scft, f0_ds05)

        update_m2_kernel!(scft, N, sg, 3, 0.10, lattice)
        f0_ds10 = extract_subgrid(u_sym, N, L)
        execute_m2_scft!(scft, f0_ds10)

        scft2 = plan_m2_scft(N, sg, 3, 0.10, lattice)
        f0_ref = extract_subgrid(u_sym, N, L)
        execute_m2_scft!(scft2, f0_ref)

        @test maximum(abs.(f0_ds10 .- f0_ref)) < 1e-14
        @test maximum(abs.(f0_ds05 .- f0_ds10)) > 1e-4
    end

    @testset "M2 fwd+bwd vs M7 fwd+bwd" begin
        for (sg, name) in [(225, "Fm-3m"), (229, "Im-3m")]
            @testset "$name" begin
                ops = get_ops(sg, 3, N)
                _, ops_s = find_optimal_shift(ops, N)
                L = auto_L(ops_s)
                u_sym = make_symmetric(ops_s, N)

                f0_m2 = extract_subgrid(u_sym, N, L)
                m2 = plan_m2_scft(N, sg, 3, Δs, lattice)
                execute_m2_scft!(m2, f0_m2)

                spec_asu = calc_spectral_asu(ops_s, 3, N)
                # M7 centered uses stride-2 subgrid
                f0_m7 = extract_subgrid(u_sym, N, [2,2,2])
                m7 = plan_centered_scft(spec_asu, ops_s, N, Δs, lattice)
                execute_centered_scft!(m7, f0_m7)

                # Both should match full FFT, but via different L extractions
                f0_ref_m2 = fullgrid_reference(u_sym, N, Δs, lattice, L)
                f0_ref_m7 = fullgrid_reference(u_sym, N, Δs, lattice, [2,2,2])
                @test maximum(abs.(f0_m2 .- f0_ref_m2)) < 1e-10
                @test maximum(abs.(f0_m7 .- f0_ref_m7)) < 1e-10
            end
        end
    end
end
