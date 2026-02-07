"""
Debug: Isolate Mode B FFT and recombination issues
"""

using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using CrystallographicFFT.SpectralIndexing
using FFTW

# Simple test: Single-frequency input
N = (16, 16, 16)
ops = get_ops(47, 3, N)  # Pmmm |G|=8
L = (2, 2, 2)

# Create input: Single Fourier mode at k=(0,0,0) (DC component = sum of all elements)
u = ones(Float64, N)

# Reference FFT
fft_ref = fft(u)
println("Reference FFT at (0,0,0): $(fft_ref[1,1,1])")  # Should be prod(N) = 4096

# Pack ASU (Mode B)
real_asu = pack_asu_interleaved(u, N, ops; L=L, asu_only=true)
println("ASU block size: $(size(real_asu.dim_blocks[3][1].data))")
println("ASU block data[1,1,1]: $(real_asu.dim_blocks[3][1].data[1,1,1])")

# Plan KRFFT
spec_asu = calc_spectral_asu(ops, 3, N)
plan = plan_krfft(real_asu, spec_asu, ops)

# Execute Forward FFT
map_fft!(plan, real_asu)

println("Buffer after FFT (first 5): $(plan.work_buffer[1:5])")
println("Expected FFT of ones: $(fft(ones(ComplexF64, 8,8,8))[1,1,1])")  # Should be 512

# Recombine
result = plan.recombination_map * plan.work_buffer
println("Result size: $(length(result))")

# Find DC component (h = [0,0,0])
for (h_idx, pt) in enumerate(spec_asu.points)
    h = get_k_vector(spec_asu, h_idx)
    if all(h .== 0)
        println("h=$h (idx=$h_idx): result=$(result[h_idx]), ref=$(fft_ref[1,1,1])")
        
        # Debug the recombination row
        row = plan.recombination_map[h_idx, :]
        nz = findnz(row)
        println("  Recomb row: $(length(nz[1])) non-zeros")
        for (j, v) in zip(nz...)
            println("    J=$j, V=$v")
        end
        break
    end
end
