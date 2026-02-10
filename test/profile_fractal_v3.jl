# Profile fractal KRFFT v3 step-by-step
# Usage: julia --project=test test/profile_fractal_v3.jl
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_fractal_krfft_v3, ASUOnlyPlan
using FFTW
using Statistics
using LinearAlgebra: mul!

function profile_v3(plan::ASUOnlyPlan, u::AbstractArray{<:Real}; nw=5, nt=30)
    pool = plan.asu_pool
    u_flat = vec(u)

    # --- Step 1a: Trivial gather ---
    tu = plan.trivial_u_idx
    tp = plan.trivial_pool_idx
    f_trivial = () -> begin
        @inbounds @simd for i in eachindex(tu)
            pool[tp[i]] = complex(u_flat[tu[i]])
        end
    end

    # --- Step 1b: Fused DFT ---
    f_fused = () -> begin
        @inbounds for fl in plan.fused_leaves
            poff = fl.pool_offset
            src = fl.u_src
            K = fl.dft_kernel
            n_asu = size(K, 1)
            vol = size(K, 2)
            for a in 1:n_asu
                val = zero(ComplexF64)
                for n in 1:vol
                    val += K[a, n] * u_flat[src[n]]
                end
                pool[poff + a] = val
            end
        end
    end

    # --- Step 1c: Large leaves FFTW ---
    f_fftw = () -> begin
        for li in eachindex(plan.large_leaf_pack_src)
            dims = plan.large_leaf_dims[li]
            src = plan.large_leaf_pack_src[li]
            asu = plan.large_leaf_asu_indices[li]
            poff = plan.large_leaf_pool_offsets[li]
            tmp = plan.fft_temps[dims]
            @inbounds for i in eachindex(src)
                tmp[i] = complex(u_flat[src[i]])
            end
            result = plan.fft_plans[dims] * tmp
            @inbounds for (rank, idx) in enumerate(asu)
                pool[poff + rank] = result[idx]
            end
        end
    end

    # --- Step 2: Butterfly ---
    f_bfly = () -> begin
        for group in plan.butterfly_schedule
            ps = group.parent_pool_start
            sz = group.parent_asu_size
            @inbounds @simd for j in 1:sz
                pool[ps + j] = zero(ComplexF64)
            end
            for (si, tw) in enumerate(group.sector_twiddles)
                ci = group.sector_child_pool_idx[si]
                @inbounds @simd for j in 1:sz
                    pool[ps + j] += tw[j] * pool[ci[j]]
                end
            end
        end
    end

    # --- Step 3: Extract ---
    out = plan.output_buffer
    ei = plan.extract_pool_idx
    ep = plan.extract_phase
    f_extract = () -> begin
        @inbounds @simd for i in eachindex(out)
            out[i] = ep[i] * pool[ei[i]]
        end
    end

    function bench(f)
        for _ in 1:nw; f(); end
        median([(@elapsed f()) for _ in 1:nt]) * 1000
    end

    t_triv = bench(f_trivial)
    t_fused = bench(f_fused)
    t_fftw = bench(f_fftw)
    t_bfly = bench(f_bfly)
    t_ext = bench(f_extract)
    t_total = t_triv + t_fused + t_fftw + t_bfly + t_ext

    return (trivial=t_triv, fused=t_fused, fftw=t_fftw, butterfly=t_bfly,
            extract=t_ext, total=t_total)
end

for N_size in [256]
    N = (N_size, N_size, N_size)
    println("\n" * "="^70)
    println("N = $N_size")
    println("="^70)

    # FFT baseline
    u = randn(Float64, N)
    uc = complex(u); fp = plan_fft(uc); F = similar(uc)
    for _ in 1:5; mul!(F, fp, uc); end
    t_fft = median([(@elapsed mul!(F, fp, uc)) for _ in 1:30]) * 1000

    for (sg, name) in [(221, "Pm-3m"), (225, "Fm-3m"), (229, "Im-3m"), (200, "Pm-3")]
        ops = get_ops(sg, 3, N)
        spec = calc_spectral_asu(ops, 3, N)
        plan = plan_fractal_krfft_v3(spec, ops)

        r = profile_v3(plan, u)

        println("\n  $name |G|=$(length(ops))")
        println("    Trivial gather: $(round(r.trivial, digits=3))ms  ($(round(r.trivial/r.total*100, digits=1))%)")
        println("    Fused DFT:      $(round(r.fused, digits=3))ms  ($(round(r.fused/r.total*100, digits=1))%)")
        println("    FFTW leaves:    $(round(r.fftw, digits=3))ms  ($(round(r.fftw/r.total*100, digits=1))%)")
        println("    Butterfly:      $(round(r.butterfly, digits=3))ms  ($(round(r.butterfly/r.total*100, digits=1))%)")
        println("    Extract:        $(round(r.extract, digits=3))ms  ($(round(r.extract/r.total*100, digits=1))%)")
        println("    ─────────────────────────────────────────")
        println("    Total:          $(round(r.total, digits=3))ms  ← $(round(t_fft/r.total, digits=1))× vs FFT $(round(t_fft, digits=2))ms")

        # Stats
        n_fused = length(plan.fused_leaves)
        fused_ops = sum(size(fl.dft_kernel, 1) * size(fl.dft_kernel, 2) for fl in plan.fused_leaves; init=0)
        n_large = length(plan.large_leaf_pack_src)
        n_bfly = sum(g.parent_asu_size * length(g.sector_twiddles) for g in plan.butterfly_schedule; init=0)
        println("    [Stats: $(length(plan.trivial_u_idx)) trivial, $n_fused fused ($fused_ops ops), $n_large FFTW, $n_bfly bfly ops]")
    end
end
