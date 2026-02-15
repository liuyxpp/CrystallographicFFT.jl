# Run ALL original test files for comprehensive regression.
# Usage: julia --project=test test/full/run_all.jl

using Test
using CrystallographicFFT

@testset "CrystallographicFFT.jl (FULL)" begin
    for f in [
        "test_asu.jl", "test_packing.jl", "test_cfft.jl",
        "test_selective_g0.jl", "test_g0_asu.jl", "test_g0_asu_backward.jl",
        "test_centering_fold.jl", "test_m7_scft.jl", "test_q_fused.jl",
        "test_centered_backward.jl", "test_centered_scft.jl", "test_m2_scft.jl",
    ]
        @testset "$f" begin
            include(joinpath(@__DIR__, f))
        end
    end
end
