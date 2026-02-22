# Changelog

All notable changes to CrystallographicFFT.jl are documented in this file.

## [v0.3.0] — 2026-02-22

### Added
- **Device-agnostic architecture** via KernelAbstractions.jl — unified CPU/GPU compute pipeline with new `types.jl`, `kernels.jl`, `planning.jl`, and `execute.jl` modules.
- **CUDA GPU support** as a weak-dependency extension (`ext/CUDAExt.jl`); supports both General (M2) and Centered (M7) paths.
- **R2C FFT transforms**: `rcfft!`/`ircfft!` using `rfft`/`irfft` for all 230 space groups. New types: `RCFFTPlan`, `IRCFFTPlan`, `GeneralRCFFTPairPlan`.
- **Star mapping API**: `SubgridStarMap`, `build_subgrid_star_map`, `expand_stars!`, `compress_stars!` for symmetry-orbit conversions on the subgrid.
- **Stress helper**: `cfft_kk_orbsum(plan, lattice)` for orbit-weighted k⊗k sums.
- New dependencies: `Adapt.jl` v4, `CUDA.jl` v5 (weak dep).
- Documentation pages: `39-rcfft.md`, `42-gpu.md`.
- Test files: `test_cuda_ext.jl`, `test_rcfft.jl`, `test_rcfft_gpu.jl`, `bench_rcfft.jl`, `bench_gpu.jl`.

### Changed
- `recommended_N` rewritten to use dynamic `auto_L`-based divisor computation (with caching) instead of static `optimal_L` lookup; correctly handles Ia-3d and other complex groups.
- `calc_spectral_asu` inner loop refactored to zero-allocation using `NTuple` flat rotation matrices and pre-allocated worklist.
- `runtests.jl` substantially refactored to adapt to the device-agnostic API.
- Legacy modules (`KRFFT`, `QFusedKRFFT`, `DiffusionSolver`, etc.) archived to `src/archived/`.

### Optimized
- `build_subgrid_star_map`: 31× speedup for Ia-3d N=128 (22 s / 8.4 GB → 0.7 s / 4.7 MB).
- `plan_rcfft_pair`: 6× speedup for Ia-3d N=64 (5.5 s → 0.9 s).
- CUDA path: removed unnecessary syncs, enabled batched CuFFT.

### Fixed
- Minor bug in alive mask computation.

## [v0.2.0] — 2026-02-18

Initial public release with CPU-only CFFT/ICFFT for all 230 space groups.
