using CrystallographicFFT.SymmetryOps: get_ops
using FFTW, Random, LinearAlgebra

function make_symmetric(u, ops, N)
    u2 = copy(u)
    for op in ops
        R = round.(Int, op.R)
        for ci in CartesianIndices(Tuple(N))
            x = collect(Tuple(ci)) .- 1
            x2 = mod.(R * x .+ round.(Int, op.t), N) .+ 1
            u2[ci] = u2[Tuple(x2)...] = (u2[ci] + u2[Tuple(x2)...]) / 2
        end
    end
    return u2
end

function sector_fft(u, N, M, parity)
    Y = zeros(ComplexF64, Tuple(M))
    for ci in CartesianIndices(Tuple(M))
        h1 = collect(Tuple(ci)) .- 1
        val = 0.0im
        for ci2 in CartesianIndices(Tuple(M))
            x1 = collect(Tuple(ci2)) .- 1
            x = 2 .* x1 .+ parity
            phase = sum(h1[d] * x1[d] / M[d] for d in 1:3)
            val += u[(mod.(x, N) .+ 1)...] * cispi(-2 * phase)
        end
        Y[ci] = val
    end
    return Y
end

function test_cross_sector(Y_equiv, Y_rep, R, M, label)
    err = 0.0
    for ci in CartesianIndices(Tuple(M))
        h = collect(Tuple(ci)) .- 1
        h_rot = mod.(R' * h, M) .+ 1
        err = max(err, abs(Y_equiv[ci] - Y_rep[h_rot...]))
    end
    status = err < 1e-10 ? "✓" : "✗"
    println("  $label: err = $(round(err, sigdigits=3)) ($status)")
    return err
end

function test_self_symmetry(Y, R, M, label)
    err = 0.0
    for ci in CartesianIndices(Tuple(M))
        h = collect(Tuple(ci)) .- 1
        h_rot = mod.(R' * h, M) .+ 1
        err = max(err, abs(Y[ci] - Y[h_rot...]))
    end
    status = err < 1e-10 ? "✓" : "✗"
    println("  $label: err = $(round(err, sigdigits=3)) ($status)")
    return err
end

function main()
    N = [8, 8, 8]
    M = [4, 4, 4]
    Random.seed!(42)
    R_inv = [-1 0 0; 0 -1 0; 0 0 -1]
    R_sigxy = [-1 0 0; 0 -1 0; 0 0 1]

    # === Test 1: P-1 with STANDARD grid (b=0) ===
    println("=== P-1 (SG 2): standard grid (b=0) ===")
    ops = get_ops(2, 3, Tuple(N))
    u = make_symmetric(randn(Tuple(N)...), ops, N)

    Y000 = sector_fft(u, N, M, [0, 0, 0])
    Y111 = sector_fft(u, N, M, [1, 1, 1])

    test_cross_sector(Y111, Y000, R_inv, M, "Y_111 = Y_000(-h) [cross-sector]")
    test_self_symmetry(Y000, R_inv, M, "Y_000(-h) = Y_000(h) [self-sym of 000]")
    test_self_symmetry(Y111, R_inv, M, "Y_111(-h) = Y_111(h) [self-sym of 111]")

    # === Test 2: P-1 with SHIFTED grid (b=1/2) ===
    println("\n=== P-1: shifted grid (b=1/2) ===")
    # Data satisfying f(-x+1 mod N) = f(x) instead of f(-x) = f(x)
    u_sh = randn(Tuple(N)...)
    for ci in CartesianIndices(Tuple(N))
        x = collect(Tuple(ci)) .- 1
        x2 = mod.(-x .+ 1, N) .+ 1  # shifted inversion: x → -x+1
        u_sh[ci] = u_sh[Tuple(x2)...] = (u_sh[ci] + u_sh[Tuple(x2)...]) / 2
    end

    Y000s = sector_fft(u_sh, N, M, [0, 0, 0])
    Y111s = sector_fft(u_sh, N, M, [1, 1, 1])

    test_cross_sector(Y111s, Y000s, R_inv, M, "Y_111 = Y_000(-h) [cross-sector]")
    test_self_symmetry(Y000s, R_inv, M, "Y_000(-h) = Y_000(h) [self-sym of 000]")

    # === Test 3: Pmmm with standard grid ===
    println("\n=== Pmmm (SG 47): standard grid (b=0) ===")
    ops2 = get_ops(47, 3, Tuple(N))
    u2 = make_symmetric(randn(Tuple(N)...), ops2, N)

    Y000_p = sector_fft(u2, N, M, [0, 0, 0])
    Y110_p = sector_fft(u2, N, M, [1, 1, 0])

    test_cross_sector(Y110_p, Y000_p, R_sigxy, M, "Y_110 = Y_000(R_β h) [cross]")
    test_self_symmetry(Y000_p, R_sigxy, M, "Y_000(R_β h) = Y_000(h) [self-sym]")

    # === Conclusion ===
    println("\n=== CONCLUSION ===")
    println("b=0 grid: reflections create SELF-SYMMETRY within sectors")
    println("b=1/2 grid: reflections create CROSS-SECTOR equivalence")
    println("→ The shift requires DIFFERENT DATA, not just relabeled ops!")
end

main()
