module CrystallographicFFT

# ── Shared infrastructure (unchanged) ──
include("symmetry_ops.jl")
using .SymmetryOps
include("asu.jl")
using .ASU
include("spectral_indexing.jl")
using .SpectralIndexing
include("matrix_q.jl")
using .MatrixQ
include("optimal_L.jl")

# ── Device-agnostic mainline code ──
using KernelAbstractions
using AbstractFFTs
include("types.jl")
include("kernels.jl")
include("planning.jl")
include("execute.jl")
include("cfft_api.jl")
using .CFFTApi

# ── Public API ──
export AbstractCFFTPlan, AbstractCFFTPairPlan
export GeneralCFFTPairPlan, CenteredCFFTPairPlan
export CFFTPlan, ICFFTPlan
export plan_cfft, plan_icfft, plan_cfft_pair
export cfft!, icfft!
export RCFFTPlan, IRCFFTPlan, GeneralRCFFTPairPlan
export plan_rcfft, plan_ircfft, plan_rcfft_pair
export rcfft!, ircfft!
export make_diffusion_kernel, update_diffusion_kernel!
export cfft_k2, cfft_kk_orbsum
export subgrid_size, fullgrid_size, stride_factors, cfft_asu_size
export subgrid_to_fullgrid!, fullgrid_to_subgrid!
export SubgridStarMap, build_subgrid_star_map, expand_stars!, compress_stars!
export recommended_N, group_order

end
