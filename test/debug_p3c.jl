#!/usr/bin/env julia
"""
Test P3c recursive KRFFT correctly, using the working symmetrization from bench_krfft.jl.
"""

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT: RecursiveKRFFTPlan, plan_krfft_recursive,
    execute_recursive_krfft!, plan_krfft, execute_krfft!, fft_reconstruct!
using FFTW
using LinearAlgebra: mul!

function make_symmetric_field(ops, N)
    u = rand(N...)
    u_sym = zeros(N...)
    for op in ops
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(op.R * x .+ op.t, collect(N)) .+ 1
            u_sym[idx] += u[x2...]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

function test_p3c_standalone()
    println("="^60)
    println("Test 1: Standalone P3c Reconstruction (pure math, no CrystallographicFFT)")
    println("="^60)
    
    for M in [8, 16, 32]
        M2 = M ÷ 2
        
        f = rand(M, M, M)
        # p3 symmetrize
        fs = zeros(M, M, M)
        for x in 1:M, y in 1:M, z in 1:M
            fs[x,y,z] = (f[x,y,z] + f[y,z,x] + f[z,x,y]) / 3
        end
        
        # Sub-sub-grids
        sub_data = Array{Float64,3}[]
        for l in 0:1, m in 0:1, n in 0:1
            push!(sub_data, fs[n+1:2:end, m+1:2:end, l+1:2:end])
        end
        Fss = [fft(ComplexF64.(s)) for s in sub_data]
        G = fft(ComplexF64.(fs))
        
        # P3c reduced reconstruction
        G_recon = zeros(ComplexF64, M, M, M)
        for hz in 0:M-1, hy in 0:M-1, hx in 0:M-1
            hxt = mod(hx, M2); hyt = mod(hy, M2); hzt = mod(hz, M2)
            val = Fss[1][hxt+1,hyt+1,hzt+1]  # F_000
            val += cispi(-2*hx/M) * Fss[5][hyt+1,hzt+1,hxt+1]  # F_100 = F_001(hy,hz,hx)
            val += cispi(-2*hy/M) * Fss[5][hzt+1,hxt+1,hyt+1]  # F_010 = F_001(hz,hx,hy)
            val += cispi(-2*hz/M) * Fss[5][hxt+1,hyt+1,hzt+1]  # F_001
            val += cispi(-2*(hx+hy)/M) * Fss[4][hxt+1,hyt+1,hzt+1]  # F_110
            val += cispi(-2*(hx+hz)/M) * Fss[4][hzt+1,hxt+1,hyt+1]  # F_101 = F_110(hz,hx,hy)
            val += cispi(-2*(hy+hz)/M) * Fss[4][hyt+1,hzt+1,hxt+1]  # F_011 = F_110(hy,hz,hx)
            val += cispi(-2*(hx+hy+hz)/M) * Fss[8][hxt+1,hyt+1,hzt+1]  # F_111
            G_recon[hx+1,hy+1,hz+1] = val
        end
        
        err = maximum(abs.(G_recon - G))
        println("  M=$M: P3c recon err = $(round(err, sigdigits=3))  $(err < 1e-10 ? "✓" : "✗")")
    end
end

function test_p3c_integration()
    println("\n" * "="^60)
    println("Test 2: P3c via RecursiveKRFFTPlan (SG 221 Pm-3m, N=16)")
    println("="^60)
    
    sg = 221
    N_tuple = (16, 16, 16)
    dim = 3
    
    ops = get_ops(sg, dim, N_tuple)
    _, ops_s = find_optimal_shift(ops, N_tuple)
    spec = calc_spectral_asu(ops_s, dim, N_tuple)
    n_spec = length(spec.points)
    
    println("  |G|=$(length(ops_s)), n_spec=$n_spec")
    
    # Create symmetric field
    u = make_symmetric_field(ops_s, N_tuple)
    
    # Full FFT reference
    G_full = fft(ComplexF64.(u))
    
    # Extract reference spectral ASU
    ref = zeros(ComplexF64, n_spec)
    for h_idx in 1:n_spec
        h = get_k_vector(spec, h_idx)
        h1 = [mod(h[d], N_tuple[d]) for d in 1:dim]
        ref[h_idx] = G_full[h1[1]+1, h1[2]+1, h1[3]+1]
    end
    
    # Phase 1 KRFFT
    plan1 = plan_krfft(spec, ops_s)
    L = [plan1.grid_N[d] ÷ plan1.subgrid_dims[d] for d in 1:dim]
    f0 = u[1:L[1]:end, 1:L[2]:end, 1:L[3]:end]
    plan1.input_buffer .= vec(ComplexF64.(f0))
    spec1 = copy(fft_reconstruct!(plan1))
    err1 = maximum(abs.(spec1 - ref))
    println("  Phase 1 error: $(round(err1, sigdigits=3))")
    
    # Recursive KRFFT
    try
        plan_r = plan_krfft_recursive(spec, ops_s)
        spec_r = execute_recursive_krfft!(plan_r, spec, u)
        err_r = maximum(abs.(spec_r - ref))
        println("  Recursive error: $(round(err_r, sigdigits=3))")
        
        if err_r < 1e-8
            println("  ✓ PASS")
        else
            println("  ✗ FAIL")
            worst = argmax(abs.(spec_r .- ref))
            h = get_k_vector(spec, worst)
            println("    Worst h=$h: ref=$(round(ref[worst], sigdigits=4)), got=$(round(spec_r[worst], sigdigits=4))")
        end
    catch e
        println("  Error: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

function test_p3c_subgrid_only()
    println("\n" * "="^60)  
    println("Test 3: P3c on subgrid only (skip A8 reconstruction)")
    println("="^60)
    
    # Just verify that pack_p3c + fft_p3c + reconstruct_g0 correctly
    # reproduces a single subgrid FFT.
    
    sg = 221
    N_tuple = (16, 16, 16)
    dim = 3
    
    ops = get_ops(sg, dim, N_tuple)
    _, ops_s = find_optimal_shift(ops, N_tuple)
    spec = calc_spectral_asu(ops_s, dim, N_tuple)
    
    u = make_symmetric_field(ops_s, N_tuple)
    
    # Extract A8 subgrid_0 manually
    M = N_tuple .÷ 2
    f0 = u[1:2:end, 1:2:end, 1:2:end]
    G0_ref = fft(ComplexF64.(f0))
    
    # Use RecursiveKRFFTPlan just for pack + fft + G0 reconstruction
    plan_r = plan_krfft_recursive(spec, ops_s)
    
    # Pack
    CrystallographicFFT.KRFFT.pack_p3c!(plan_r, u)
    
    # FFT
    CrystallographicFFT.KRFFT.fft_p3c!(plan_r)
    
    # Reconstruct G0 at each point and compare
    max_err = 0.0
    for hz in 0:M[3]-1, hy in 0:M[2]-1, hx in 0:M[1]-1
        g0_val = CrystallographicFFT.KRFFT.reconstruct_g0_at(plan_r, hx, hy, hz)
        ref_val = G0_ref[hx+1, hy+1, hz+1]
        err = abs(g0_val - ref_val)
        max_err = max(max_err, err)
    end
    
    println("  G0 reconstruction max error: $(round(max_err, sigdigits=3))")
    if max_err < 1e-10
        println("  ✓ PASS")
    else
        println("  ✗ FAIL")
        # Find worst point
        for hz in 0:M[3]-1, hy in 0:M[2]-1, hx in 0:M[1]-1
            g0_val = CrystallographicFFT.KRFFT.reconstruct_g0_at(plan_r, hx, hy, hz)
            ref_val = G0_ref[hx+1, hy+1, hz+1]
            if abs(g0_val - ref_val) > max_err * 0.9
                println("    Worst: h=($hx,$hy,$hz) ref=$(round(ref_val,sigdigits=4)) got=$(round(g0_val,sigdigits=4))")
                break
            end
        end
    end
end

test_p3c_standalone()
test_p3c_subgrid_only()
test_p3c_integration()
