# Correctness tests for staged KRFFT (Pm-3m, SG 221)
#
# Tests:
#   1. F₀₀₀ reconstruction: execute_staged_full! result == fft(f₀₀₀)
#   2. Full spectrum via lookup_full_spectrum: spot-check against fft(u_sym)
#   3. Consistency: staged spectral ASU == G0 ASU spectral ASU
#
# Usage: julia --project=test test/test_staged_pm3m.jl

using Test
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu, get_k_vector
using CrystallographicFFT.KRFFT: plan_krfft_g0asu, execute_g0asu_krfft!
using FFTW
using LinearAlgebra: mul!

FFTW.set_num_threads(1)

include(joinpath(@__DIR__, "..", "src", "staged_krfft.jl"))

# ─── Helper ──────────────────────────────────────────────────────────────

"""Generate Pm-3m symmetric data via real-space averaging.
u_sym(x) = (1/|G|) Σ_g u(R_g x + t_g) ensures all point-group symmetries."""
function make_symmetric(N::NTuple{3,Int}, ops)
    u_raw = randn(N...)
    u_sym = zeros(N...)
    N_arr = collect(N)
    for k in 0:N[3]-1, j in 0:N[2]-1, i in 0:N[1]-1
        val = 0.0
        for op in ops
            x2 = mod.(round.(Int, op.R * [i,j,k] .+ op.t), N_arr)
            val += u_raw[x2[1]+1, x2[2]+1, x2[3]+1]
        end
        u_sym[i+1, j+1, k+1] = val / length(ops)
    end
    return u_sym
end

# ─── Tests ───────────────────────────────────────────────────────────────

@testset "Staged KRFFT Pm-3m" begin

    for N in [32, 64]
        @testset "N=$N" begin
            Nt = (N, N, N)
            N2 = N ÷ 2

            ops = get_ops(221, 3, Nt)
            _, ops_s = find_optimal_shift(ops, Nt)
            u = make_symmetric(Nt, ops_s)

            # ── Plan ──
            fp = plan_staged_pm3m_full(Nt)

            # ── Test 1: F₀₀₀ correctness ──
            @testset "F₀₀₀ vs fft(f₀₀₀)" begin
                F_staged = execute_staged_full!(fp, u)

                # Reference: extract f₀₀₀ = u[2x₁+1] and FFT it
                pack_a8!(fp.staged, u)
                ref_F000 = fft(copy(fp.staged.a8_buf))

                max_err = maximum(abs.(F_staged .- ref_F000))
                @test max_err < 1e-10
                @info "N=$N F₀₀₀ max error: $max_err"
            end

            # ── Test 2: lookup_full_spectrum vs fft(u_sym) ──
            @testset "lookup_full_spectrum" begin
                # Re-run forward pass
                execute_staged_full!(fp, u)

                # Reference: full FFT of u
                F_ref = fft(complex(u))

                # Spot-check 50 random h-vectors
                max_err = 0.0
                for _ in 1:50
                    hx = rand(0:N-1); hy = rand(0:N-1); hz = rand(0:N-1)
                    val_staged = lookup_full_spectrum(fp.staged, fp.output, hx, hy, hz)
                    val_ref = F_ref[hx+1, hy+1, hz+1]
                    max_err = max(max_err, abs(val_staged - val_ref))
                end
                @test max_err < 1e-10
                @info "N=$N lookup_full_spectrum max error: $max_err"
            end

            # ── Test 3: Consistency with G0 ASU ──
            @testset "vs G0 ASU" begin
                spec = calc_spectral_asu(ops_s, 3, Nt)
                plan_g0 = plan_krfft_g0asu(spec, ops_s)

                # G0 ASU result
                F_g0 = copy(execute_g0asu_krfft!(plan_g0, spec, u))

                # Staged: query same spectral ASU points
                execute_staged_full!(fp, u)
                n_spec = length(spec.points)
                max_err = 0.0
                for h_idx in 1:n_spec
                    h_vec = get_k_vector(spec, h_idx)
                    val_staged = lookup_full_spectrum(fp.staged, fp.output, h_vec...)
                    max_err = max(max_err, abs(val_staged - F_g0[h_idx]))
                end
                @test max_err < 1e-10
                @info "N=$N staged vs G0 ASU max error: $max_err"
            end
        end
    end
end
