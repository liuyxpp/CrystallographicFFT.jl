module CrystallographicFFT

include("symmetry_ops.jl")
using .SymmetryOps
include("asu.jl")
using .ASU
include("spectral_indexing.jl")
using .SpectralIndexing
include("matrix_q.jl")
using .MatrixQ
include("cfft_plan.jl")
include("krfft.jl")
using .KRFFT
include("q_fused_krfft.jl")
using .QFusedKRFFT
include("diffusion_solver.jl")
using .DiffusionSolver
include("optimal_L.jl")
include("execution.jl")
include("cfft_api.jl")
using .CFFTApi

# ---- Public API ----
export AbstractCFFTPlan, AbstractCFFTPairPlan
export GeneralCFFTPairPlan, CenteredCFFTPairPlan
export CFFTPlan, ICFFTPlan
export plan_cfft, plan_icfft, plan_cfft_pair
export cfft!, icfft!
export make_diffusion_kernel, update_diffusion_kernel!
export cfft_k2
export subgrid_size, fullgrid_size, stride_factors, cfft_asu_size
export subgrid_to_fullgrid!, fullgrid_to_subgrid!
export recommended_N, group_order

end


