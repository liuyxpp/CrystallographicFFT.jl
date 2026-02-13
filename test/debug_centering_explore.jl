#=
Centering Exploration: Can F000 and F111 be merged under F-centering?

For Fm-3m, the P3c decomposition gives 4 sub-FFTs:
  F000: u(4n)           parity (0,0,0)
  F001: u(4n + (0,0,2)) parity (0,0,1)  ← extinct under F-centering
  F110: u(4n + (2,2,0)) parity (1,1,0)  ← extinct under F-centering
  F111: u(4n + (2,2,2)) parity (1,1,1)

Question: Does F-centering create any relationship between F000 and F111
that would allow computing both from a single FFT?
=#

using FFTW
using LinearAlgebra

function test_centering_relations(N::Int)
    @assert N % 8 == 0 "N must be divisible by 8 for this test"
    M2 = N ÷ 4  # P3c sub-sub-grid size

    println("=" ^ 60)
    println("N = $N, M2 = $M2")
    println("=" ^ 60)

    # Create random symmetric (Fm-3m) data
    # F-centering: u(x) = u(x + τ) for τ ∈ {(0,N/2,N/2), (N/2,0,N/2), (N/2,N/2,0)}
    u = rand(N, N, N)
    # Enforce F-centering symmetry
    for i in 1:N, j in 1:N, k in 1:N
        τ_list = [(0, N÷2, N÷2), (N÷2, 0, N÷2), (N÷2, 0+N÷2, N÷2+0)]
        # Actually, let's symmetrize properly
    end
    # Better: generate on primitive cell and tile
    u_prim = rand(N÷2, N÷2, N÷2)
    u = zeros(N, N, N)
    # F-centering: lattice vectors for conventional cell from primitive:
    # Positions at (2i, 2j, 2k), (2i, 2j+1, 2k+1), (2i+1, 2j, 2k+1), (2i+1, 2j+1, 2k)
    # in units of N/2
    for i in 0:N÷2-1, j in 0:N÷2-1, k in 0:N÷2-1
        val = u_prim[i+1, j+1, k+1]
        # Place at all F-centering equivalent positions
        positions = [
            (2i, 2j, 2k),
            (2i, 2j + N÷2, 2k + N÷2),
            (2i + N÷2, 2j, 2k + N÷2),
            (2i + N÷2, 2j + N÷2, 2k),
        ]
        for (pi, pj, pk) in positions
            u[mod(pi, N)+1, mod(pj, N)+1, mod(pk, N)+1] = val
        end
    end

    # Verify F-centering
    τs = [(0, N÷2, N÷2), (N÷2, 0, N÷2), (N÷2, N÷2, 0)]
    for τ in τs
        err = 0.0
        for i in 1:N, j in 1:N, k in 1:N
            ii = mod(i-1+τ[1], N)+1
            jj = mod(j-1+τ[2], N)+1
            kk = mod(k-1+τ[3], N)+1
            err = max(err, abs(u[i,j,k] - u[ii, jj, kk]))
        end
        println("F-centering τ=$τ: max error = $err")
    end

    # Extract P3c sub-grids
    buf000 = zeros(ComplexF64, M2, M2, M2)
    buf001 = zeros(ComplexF64, M2, M2, M2)
    buf110 = zeros(ComplexF64, M2, M2, M2)
    buf111 = zeros(ComplexF64, M2, M2, M2)

    for k in 1:M2, j in 1:M2, i in 1:M2
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000[i,j,k] = u[ii+1, jj+1, kk+1]
        buf001[i,j,k] = u[ii+1, jj+1, kk+3]
        buf110[i,j,k] = u[ii+3, jj+3, kk+1]
        buf111[i,j,k] = u[ii+3, jj+3, kk+3]
    end

    # Check: are F001 and F110 identically zero after FFT?
    F000 = fft(buf000)
    F001 = fft(buf001)
    F110 = fft(buf110)
    F111 = fft(buf111)

    println("\nFFT magnitudes:")
    println("  |F000|_max = $(maximum(abs, F000))")
    println("  |F001|_max = $(maximum(abs, F001))")
    println("  |F110|_max = $(maximum(abs, F110))")
    println("  |F111|_max = $(maximum(abs, F111))")

    # Check: what's the relationship between buf000 and buf111?
    println("\nRelationship between buf000 and buf111:")

    # Test 1: Are they equal?
    diff_direct = maximum(abs, buf000 .- buf111)
    println("  max|buf000 - buf111| = $diff_direct")

    # Test 2: Is buf111 a shifted version of buf000?
    for shift in [(0,0,0), (M2÷2, M2÷2, M2÷2), (M2÷2, M2÷2, 0),
                  (0, M2÷2, M2÷2), (M2÷2, 0, M2÷2)]
        shifted = zeros(ComplexF64, M2, M2, M2)
        for k in 1:M2, j in 1:M2, i in 1:M2
            si = mod(i-1+shift[1], M2)+1
            sj = mod(j-1+shift[2], M2)+1
            sk = mod(k-1+shift[3], M2)+1
            shifted[i,j,k] = buf000[si, sj, sk]
        end
        diff = maximum(abs, shifted .- buf111)
        if diff < 1e-10
            println("  ✅ buf111 = circshift(buf000, $shift) ! diff=$diff")
        else
            println("  ❌ buf111 ≠ circshift(buf000, $shift), diff=$diff")
        end
    end

    # Test 3: Internal periodicity of buf000 under F-centering
    println("\nInternal periodicity of buf000:")
    for shift in [(M2÷2, M2÷2, 0), (0, M2÷2, M2÷2), (M2÷2, 0, M2÷2)]
        shifted = zeros(ComplexF64, M2, M2, M2)
        for k in 1:M2, j in 1:M2, i in 1:M2
            si = mod(i-1+shift[1], M2)+1
            sj = mod(j-1+shift[2], M2)+1
            sk = mod(k-1+shift[3], M2)+1
            shifted[i,j,k] = buf000[si, sj, sk]
        end
        diff = maximum(abs, shifted .- buf000)
        println("  shift=$shift: max|buf000 - circshift(buf000, shift)| = $diff")
    end

    println("\nInternal periodicity of buf111:")
    for shift in [(M2÷2, M2÷2, 0), (0, M2÷2, M2÷2), (M2÷2, 0, M2÷2)]
        shifted = zeros(ComplexF64, M2, M2, M2)
        for k in 1:M2, j in 1:M2, i in 1:M2
            si = mod(i-1+shift[1], M2)+1
            sj = mod(j-1+shift[2], M2)+1
            sk = mod(k-1+shift[3], M2)+1
            shifted[i,j,k] = buf111[si, sj, sk]
        end
        diff = maximum(abs, shifted .- buf111)
        println("  shift=$shift: max|buf111 - circshift(buf111, shift)| = $diff")
    end

    # Test 4: Spectral extinction pattern of F000
    println("\nSpectral extinction of F000 (non-zero fraction):")
    nnz_000 = count(x -> abs(x) > 1e-10 * maximum(abs, F000), F000)
    println("  Non-zero: $nnz_000 / $(length(F000)) = $(round(nnz_000/length(F000)*100, digits=1))%")

    nnz_111 = count(x -> abs(x) > 1e-10 * maximum(abs, F111), F111)
    println("  Non-zero F111: $nnz_111 / $(length(F111)) = $(round(nnz_111/length(F111)*100, digits=1))%")

    # Test 5: Which (qx,qy,qz) parities survive in F000?
    println("\nF000 spectral parity pattern:")
    for px in 0:1, py in 0:1, pz in 0:1
        count_nz = 0
        count_total = 0
        for qz in 0:M2-1, qy in 0:M2-1, qx in 0:M2-1
            if mod(qx,2) == px && mod(qy,2) == py && mod(qz,2) == pz
                count_total += 1
                if abs(F000[qx+1,qy+1,qz+1]) > 1e-10 * maximum(abs, F000)
                    count_nz += 1
                end
            end
        end
        if count_total > 0
            frac = round(count_nz/count_total*100, digits=0)
            println("  parity ($px,$py,$pz): $count_nz/$count_total non-zero ($frac%)")
        end
    end

    # Test 6: Similarly I-centering
    println("\n--- I-centering comparison ---")
    u_I = rand(N, N, N)
    # I-centering: u(x) = u(x + (N/2,N/2,N/2))
    for i in 1:N, j in 1:N, k in 1:N
        ii = mod(i-1+N÷2, N)+1
        jj = mod(j-1+N÷2, N)+1
        kk = mod(k-1+N÷2, N)+1
        u_I[ii, jj, kk] = u_I[i, j, k]
    end

    buf000_I = zeros(ComplexF64, M2, M2, M2)
    buf001_I = zeros(ComplexF64, M2, M2, M2)
    buf110_I = zeros(ComplexF64, M2, M2, M2)
    buf111_I = zeros(ComplexF64, M2, M2, M2)

    for k in 1:M2, j in 1:M2, i in 1:M2
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000_I[i,j,k] = u_I[ii+1, jj+1, kk+1]
        buf001_I[i,j,k] = u_I[ii+1, jj+1, kk+3]
        buf110_I[i,j,k] = u_I[ii+3, jj+3, kk+1]
        buf111_I[i,j,k] = u_I[ii+3, jj+3, kk+3]
    end

    F001_I = fft(buf001_I)
    F111_I = fft(buf111_I)
    println("I-centering |F001|_max = $(maximum(abs, F001_I))")
    println("I-centering |F111|_max = $(maximum(abs, F111_I))")

    return nothing
end

for N in [16, 32, 64]
    test_centering_relations(N)
    println()
end
