"""
Debug: Trace Mode B reconstruction for a specific non-DC frequency
"""

using CrystallographicFFT
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using CrystallographicFFT.KRFFT
using CrystallographicFFT.SpectralIndexing
using FFTW
using SparseArrays

# Simple test: Single-frequency input
N = (16, 16, 16)
ops = get_ops(47, 3, N)  # Pmmm |G|=8
L = (2, 2, 2)

# Create test data - single Fourier mode at k=(1,0,0)
u = zeros(ComplexF64, N)
for n in CartesianIndices(u)
    idx = Tuple(n) .- 1  # 0-based
    phase = sum([1, 0, 0] .* idx ./ collect(N))
    u[n] = exp(2π * im * phase)
end
u = real.(u)  # Keep real part only

# Reference FFT
fft_ref = fft(u)
println("Reference FFT at (1,0,0): $(fft_ref[2,1,1])")  # 1-based indexing
println("Reference FFT at (-1,0,0) = (15,0,0): $(fft_ref[16,1,1])")

# Pack ASU (Mode B)
real_asu = pack_asu_interleaved(u, N, ops; L=L, asu_only=true)
println("ASU block size: $(size(real_asu.dim_blocks[3][1].data))")

# Plan KRFFT
spec_asu = calc_spectral_asu(ops, 3, N)
plan = plan_krfft(real_asu, spec_asu, ops)

# Execute Forward FFT
map_fft!(plan, real_asu)

# Debug: Check subgrid FFT directly
subgrid_fft = fft(real_asu.dim_blocks[3][1].data)
println("Subgrid FFT at (1,0,0): $(subgrid_fft[2,1,1])")  # 8x8x8 grid
println("Subgrid FFT at (0,0,0): $(subgrid_fft[1,1,1])")

# Check buffer content matches subgrid FFT
println("Buffer[1]: $(plan.work_buffer[1]) (should match subgrid_fft[1,1,1] = $(subgrid_fft[1,1,1]))")
println("Buffer[2]: $(plan.work_buffer[2]) (should match subgrid_fft[2,1,1] = $(subgrid_fft[2,1,1]))")

# Trace reconstruction for h=(1,0,0)
println("\n=== Tracing h=[1,0,0] ===")
for (h_idx, pt) in enumerate(spec_asu.points)
    h = get_k_vector(spec_asu, h_idx)
    if h == [1, 0, 0]
        println("Found h=$h at idx=$h_idx")
        
        # Get reconstruction row
        row = plan.recombination_map[h_idx, :]
        nz_indices, nz_vals = findnz(row)
        
        result_val = 0.0 + 0.0im
        for (j, v) in zip(nz_indices, nz_vals)
            println("  J=$j, weight=$v, buffer[$j]=$(plan.work_buffer[j])")
            result_val += v * plan.work_buffer[j]
        end
        
        println("Reconstructed: $result_val")
        println("Reference: $(fft_ref[2,1,1])")
        break
    end
end
