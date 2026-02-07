"""
Debug: Trace Mode B reconstruction for problematic frequency h=[-8,-8,-8]
"""

using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using CrystallographicFFT.SpectralIndexing
using FFTW
using SparseArrays

N = (16, 16, 16)
ops = get_ops(47, 3, N)  # Pmmm |G|=8
L = (2, 2, 2)

# Create Pmmm-symmetric data
u_raw = rand(Float64, N)
u = zeros(Float64, N)

for i in CartesianIndices(u)
    idx = collect(Tuple(i) .- 1)
    val = 0.0
    for op in ops
        new_idx = mod.(op.R * idx, collect(N))
        new_i = CartesianIndex(Tuple(new_idx .+ 1)...)
        val += u_raw[new_i]
    end
    u[i] = val / length(ops)
end

# Reference FFT
fft_ref = fft(u)

# Pack ASU
real_asu = pack_asu_interleaved(u, N, ops; L=L, asu_only=true)
spec_asu = calc_spectral_asu(ops, 3, N)
plan = plan_krfft(real_asu, spec_asu, ops)

# Execute FFT
map_fft!(plan, real_asu)

# Get subgrid FFT directly for verification
subgrid_fft = fft(Complex.(real_asu.dim_blocks[3][1].data))
println("Subgrid FFT [0,0,0]: $(subgrid_fft[1,1,1])")

# Find h=[-8,-8,-8] in spectral ASU
for (h_idx, pt) in enumerate(spec_asu.points)
    h = get_k_vector(spec_asu, h_idx)
    if h == [-8, -8, -8]
        println("\n=== h=[-8,-8,-8] at idx=$h_idx ===")
        
        # Reference
        ref_idx = Tuple(mod.(h, N) .+ 1)
        println("FFTW index: $ref_idx, value: $(fft_ref[ref_idx...])")
        
        # Get recombination row
        row = plan.recombination_map[h_idx, :]
        nz_indices, nz_vals = findnz(row)
        
        println("\nRecombination entries:")
        result_val = 0.0 + 0.0im
        for (j, v) in zip(nz_indices, nz_vals)
            buf_val = plan.work_buffer[j]
            println("  J=$j, weight=$v, buffer[$j]=$buf_val, contrib=$(v * buf_val)")
            result_val += v * buf_val
        end
        
        println("\nReconstruced sum: $result_val")
        println("Reference: $(fft_ref[ref_idx...])")
        println("Error: $(abs(result_val - fft_ref[ref_idx...]))")
        
        # Manual verification using KRFFT V Eq. 27
        println("\n--- Manual KRFFT V Eq. 27 verification ---")
        N_sub = Tuple(N[d] ÷ L[d] for d in 1:3)
        manual_sum = 0.0 + 0.0im
        
        for (i, op) in enumerate(ops)
            # q = R^T * h
            q = transpose(op.R) * h
            
            # Local index in subgrid: q mod N_sub
            k_local = mod.(q, N_sub) .+ 1  # 1-based
            
            # Get subgrid FFT value
            Y_val = subgrid_fft[k_local...]
            
            # Phase: e^{-2πi h·t}
            phase_val = sum(h .* op.t ./ collect(N))
            sym_phase = exp(-2π * im * phase_val)
            
            println("  Op $i: R^T*h=$q, k_local=$k_local, Y=$Y_val, phase=$sym_phase")
            manual_sum += sym_phase * Y_val
        end
        
        println("\nManual sum: $manual_sum")
        println("Difference from ref: $(abs(manual_sum - fft_ref[ref_idx...]))")
        
        break
    end
end
