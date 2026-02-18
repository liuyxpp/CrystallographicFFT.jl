using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, detect_centering_type, CentF, CentI, CentC
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using FFTW
using Random

# ── Helpers ──────────────────────────────────────────────────────────────────

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

# ── Test 1: Centering fold + assemble = direct FFT ──────────────────────────

println("="^70)
println("TEST 1: Centering fold + assemble G₀ vs direct FFT of f₀")
println("="^70)

for (sg, name, cent) in [
    (70,  "Fddd",  CentF),
    (229, "Im-3m", CentI),
    (63,  "Cmcm",  CentC),
]
    N = (32, 32, 32)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    u_sym = make_symmetric(ops_s, N)

    M = N .÷ 2
    f0 = Float64[u_sym[2i-1,2j-1,2k-1] for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
    G0_ref = fft(f0)

    fold_plan = CrystallographicFFT.plan_centering_fold(Float64, cent, M)
    CrystallographicFFT.centering_fold!(fold_plan, f0)
    CrystallographicFFT.fft_channels!(fold_plan)

    G0_assembled = zeros(ComplexF64, M)
    CrystallographicFFT.assemble_G0!(G0_assembled, fold_plan)

    err = maximum(abs, G0_assembled .- G0_ref)
    status = err < 1e-10 ? "✓ PASS" : "✗ FAIL"
    println("  $name (SG$sg): max_err = $(round(err, sigdigits=4))  $status")
end

# ── Test 2: Fold/Unfold roundtrip ───────────────────────────────────────────

println("\n" * "="^70)
println("TEST 2: Centering fold → unfold roundtrip")
println("="^70)

for (sg, name, cent) in [
    (225, "Fm-3m", CentF),
    (229, "Im-3m", CentI),
    (63,  "Cmcm",  CentC),
]
    N = (32, 32, 32)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    u_sym = make_symmetric(ops_s, N)

    M = N .÷ 2
    f0 = Float64[u_sym[2i-1,2j-1,2k-1] for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
    f0_orig = copy(f0)

    fold_plan = CrystallographicFFT.plan_centering_fold(Float64, cent, M)
    CrystallographicFFT.centering_fold!(fold_plan, f0)
    CrystallographicFFT.fft_channels!(fold_plan)
    CrystallographicFFT.ifft_channels!(fold_plan)
    CrystallographicFFT.centering_unfold!(fold_plan, f0)

    err = maximum(abs, f0 .- f0_orig)
    status = err < 1e-12 ? "✓ PASS" : "✗ FAIL"
    println("  $name (SG$sg): max_err = $(round(err, sigdigits=4))  $status")
end

# ── Test 3: Forward spectral consistency ────────────────────────────────────

println("\n" * "="^70)
println("TEST 3: Centered forward spectral consistency")
println("="^70)

for (sg, name) in [
    (225, "Fm-3m"),
    (229, "Im-3m"),
    (63,  "Cmcm"),
    (70,  "Fddd"),
]
    N = (32, 32, 32)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, 3, N)
    u_sym = make_symmetric(ops_s, N)
    F_ref = fft(u_sym)

    fwd = CrystallographicFFT.plan_centered_forward(Float64, spec, ops_s)
    M = fwd.krfft_plan.M

    # Pack stride-2 subgrid into f0_buffer
    for k in 1:M[3], j in 1:M[2], i in 1:M[1]
        fwd.f0_buffer[i,j,k] = u_sym[2*(i-1)+1, 2*(j-1)+1, 2*(k-1)+1]
    end

    F_cfft = CrystallographicFFT.fft_reconstruct_centered!(fwd)

    max_err = 0.0
    n_spec = length(spec.points)
    for i in 1:n_spec
        h = get_k_vector(spec, i)
        fref = F_ref[mod(h[1],N[1])+1, mod(h[2],N[2])+1, mod(h[3],N[3])+1]
        max_err = max(max_err, abs(F_cfft[i] - fref))
    end

    status = max_err < 1e-8 ? "✓ PASS" : "✗ FAIL"
    println("  $name (SG$sg): n_spec=$n_spec, max_err=$(round(max_err, sigdigits=4))  $status")
end

# ── Test 4: Backward roundtrip ──────────────────────────────────────────────

println("\n" * "="^70)
println("TEST 4: Forward → Backward roundtrip")
println("="^70)

for (sg, name) in [
    (225, "Fm-3m"),
    (229, "Im-3m"),
    (63,  "Cmcm"),
    (70,  "Fddd"),
]
    N = (32, 32, 32)
    ops = get_ops(sg, 3, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, 3, N)
    u_sym = make_symmetric(ops_s, N)

    fwd = CrystallographicFFT.plan_centered_forward(Float64, spec, ops_s)
    M = fwd.krfft_plan.M
    M_vol = prod(M)

    # Pack f0
    f0_orig = zeros(Float64, M...)
    for k in 1:M[3], j in 1:M[2], i in 1:M[1]
        f0_orig[i,j,k] = u_sym[2*(i-1)+1, 2*(j-1)+1, 2*(k-1)+1]
    end
    copyto!(fwd.f0_buffer, f0_orig)

    # Forward
    F_cfft = CrystallographicFFT.fft_reconstruct_centered!(fwd)
    F_spec = copy(fwd.krfft_plan.output_buffer)

    # Backward
    bwd = CrystallographicFFT.plan_centered_backward(Float64, spec, ops_s)
    f0_out = CrystallographicFFT.execute_centered_backward!(bwd, F_spec)

    # Compare
    f0_rec = zeros(Float64, M...)
    for i in 1:M_vol
        f0_rec[i] = real(f0_out[i])
    end

    rt_err = maximum(abs, f0_rec .- f0_orig)
    status = rt_err < 1e-10 ? "✓ PASS" : "✗ FAIL"
    println("  $name (SG$sg): roundtrip_err=$(round(rt_err, sigdigits=4))  $status")
end

println("\n" * "="^70)
println("All tests completed!")
println("="^70)
