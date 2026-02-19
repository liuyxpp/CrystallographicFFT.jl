# Theory & Algorithms

CrystallographicFFT.jl is based on the **KRFFT** (Kunis–Rössler FFT) algorithm, which exploits the symmetry of crystallographic space groups to reduce the computational cost of 3D FFTs.

## Cooley–Tukey Subgrid Decomposition

The standard $N$-point DFT can be decomposed via the Cooley–Tukey identity. For a 3D grid of size $N^3$ with stride factor $L$, the subgrid size is $M = N/L$, and each frequency $\mathbf{h}$ is computed as:

```math
F(\mathbf{h}) = \sum_{\mathbf{x}_0 \in \{0,\ldots,L-1\}^3} \omega^{\mathbf{h} \cdot \mathbf{x}_0} \, Y_{\mathbf{x}_0}(\mathbf{h} \bmod M)
```

where $\omega = e^{-2\pi i / N}$ and $Y_{\mathbf{x}_0}$ is the FFT of the subgrid at shift $\mathbf{x}_0$.

The key insight: if the input field $f$ is symmetric under a space group $G$ of order $|G|$, then only **one subgrid** (the $\mathbf{x}_0 = \mathbf{0}$ subgrid) is needed, and the contributions of all $L^3$ shifts can be algebraically combined using the symmetry operations.

## Spectral Asymmetric Unit (ASU)

Under a space group $G$, the $N^3$ spectral coefficients $F(\mathbf{h})$ are not all independent. The **spectral ASU** is the set of unique representatives — one from each orbit under the point group action:

```math
n_{\text{spec}} \approx \frac{N^3}{|G|}
```

For Pm-3m ($|G| = 48$), this reduces the spectrum to $\sim N^3/48$ unique values.

The spectral ASU is computed automatically by `calc_spectral_asu`, which:
1. Enumerates all frequencies $\mathbf{h} \in [0, N)^3$
2. Groups them into orbits under the point group
3. Selects one canonical representative per orbit

## Shift Optimization

To simplify the reconstruction formula, each symmetry operation $g = \{R | \mathbf{t}\}$ is shifted by a constant vector $\mathbf{b}$ (the "magic shift"):

```math
g_{\text{shifted}} = \{R | \mathbf{t}' \}, \quad \mathbf{t}' = \mathbf{t} + (R - I)\mathbf{b}
```

The optimal $\mathbf{b}$ is chosen so that $\mathbf{t}' \cdot N$ is an integer for all operations, which eliminates irrational phase factors in the reconstruction formula. This is computed by `find_optimal_shift`.

## Forward Reconstruction

Given the subgrid FFT output $Y(\mathbf{m})$ for $\mathbf{m} \in [0, M)^3$, the spectral ASU values are reconstructed by summing over symmetry operations:

```math
F(\mathbf{h}) = \frac{1}{|G|} \sum_{g \in G} e^{-2\pi i \mathbf{h} \cdot \mathbf{t}_g / N} \, Y(R_g^{-1} \mathbf{h} \bmod M)
```

This is the core of `fft_reconstruct!`. The implementation uses two paths:
- **Diagonal path**: when $R_g^{-1}\mathbf{h} \bmod M$ maps to a unique subgrid index (simple table lookup)
- **General path**: uses sparse matrix multiplication for complex orbit structures

## Inverse Reconstruction

The backward transform reverses this process:

```math
Y(\mathbf{m}) = \sum_{\mathbf{h} \in \text{ASU}} w_{\mathbf{h}} \, e^{2\pi i \mathbf{h} \cdot \mathbf{t}_g / N} \, F(\mathbf{h})
```

where $w_{\mathbf{h}}$ accounts for the orbit multiplicity. This is followed by an inverse FFT on the subgrid to recover the real-space field.

## Automatic Stride Factor Selection

The stride factor $L$ determines the subgrid size $M = N/L$. The function `auto_L` selects the optimal $L$ for each dimension by analyzing the translational components of the symmetry operations:

- **Cubic groups**: typically $L = (2, 2, 2)$, giving $M = N/2$ and an $8\times$ volume reduction
- **Lower symmetry**: $L$ may be anisotropic, e.g., $L = (2, 2, 1)$ for tetragonal groups with fewer translational symmetries

## Real-Valued FFT Optimization

When the input field is real-valued (as in most physical applications), the spectral coefficients satisfy Hermitian symmetry: $F(-\mathbf{h}) = \overline{F(\mathbf{h})}$. The `rcfft!`/`ircfft!` API exploits this by using `rfft`/`irfft`:

- **Forward**: `rfft` outputs a half-spectrum of size $\hat{M}_1 \times M_2 \times M_3$ where $\hat{M}_1 = M_1 \div 2 + 1$
- **Backward**: inverse reconstruction fills only the half-spectrum, and `irfft` directly outputs real data

Frequencies beyond the half-spectrum boundary are recovered via Hermitian conjugation. The reconstruction table uses **signed index encoding**: positive indices access the `rfft` output directly, negative indices access the conjugate. This avoids explicit data copying or building the full complex spectrum.

## Device-Agnostic Architecture

All compute-intensive operations (reconstruction, centering fold/unfold, assembly) are implemented as [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) `@kernel` functions. This enables the same code to execute on:

- **CPU**: kernels map to optimized scalar loops
- **GPU**: kernels map to parallel CUDA thread blocks (via the CUDA.jl extension)

Plan construction (spectral ASU enumeration, table building, matrix inversion) runs on CPU. The resulting tables and buffers are transferred to the target device via `_to_device(backend, data)`. FFT plans are created on device arrays, so FFTW and CUFFT dispatch automatically through the `AbstractFFTs` interface.
