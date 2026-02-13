"""
Benchmark: centering pre-fold vs full FFT

Measures end-to-end time for `execute_centered_krfft!` compared with raw `fft()`.
"""

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using CrystallographicFFT.SpectralIndexing
using FFTW
using Statistics

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

function bench_one(sg_num, N; n_warmup=3, n_trials=20)
    u_sym = make_symmetric(sg_num, N)

    # Build plans
    plan_c = plan_krfft_centered(sg_num, N)
    fft_plan = plan_fft(zeros(ComplexF64, N...))
    u_complex = complex(u_sym)

    # Warmup
    for _ in 1:n_warmup
        execute_centered_krfft!(plan_c, u_sym)
        mul!(similar(u_complex), fft_plan, u_complex)
    end

    # Benchmark centered KRFFT
    t_centered = zeros(n_trials)
    for i in 1:n_trials
        t_centered[i] = @elapsed execute_centered_krfft!(plan_c, u_sym)
    end

    # Benchmark full FFT
    out_fft = similar(u_complex)
    t_fft = zeros(n_trials)
    for i in 1:n_trials
        t_fft[i] = @elapsed mul!(out_fft, fft_plan, u_complex)
    end

    return median(t_centered), median(t_fft)
end

function main()
    cases = [
        (229, "Im-3m",  "I", [(16,16,16), (32,32,32), (64,64,64)]),
        (225, "Fm-3m",  "F", [(16,16,16), (32,32,32), (64,64,64)]),
        (70,  "Fddd",   "F", [(16,16,16), (32,32,32), (64,64,64)]),
        (139, "I4/mmm", "I", [(16,16,16), (32,32,32), (64,64,64)]),
        (35,  "Cmm2",   "C", [(16,16,16), (32,32,32), (64,64,64)]),
        (69,  "Fmmm",   "F", [(16,16,16), (32,32,32), (64,64,64)]),
    ]

    println("=" ^ 85)
    println("Centering Pre-fold Benchmark")
    println("=" ^ 85)
    println()
    header = rpad("SG", 12) * rpad("Cent", 5) * rpad("N", 14) *
             rpad("t_centered", 14) * rpad("t_fft", 14) * rpad("speedup", 10)
    println(header)
    println("-" ^ 85)

    for (sg, name, cent, grids) in cases
        for N in grids
            t_c, t_f = bench_one(sg, N)
            speedup = t_f / t_c
            line = rpad("$sg $name", 12) * rpad(cent, 5) *
                   rpad("$(N[1])³", 14) *
                   rpad("$(round(t_c*1e6, digits=1)) μs", 14) *
                   rpad("$(round(t_f*1e6, digits=1)) μs", 14) *
                   rpad("$(round(speedup, digits=2))×", 10)
            println(line)
        end
    end
    println("=" ^ 85)
end

main()
