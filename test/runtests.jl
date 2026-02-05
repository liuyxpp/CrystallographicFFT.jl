using Test
using CrystallographicFFT

@testset "CrystallographicFFT.jl" begin
    @testset "ASU Construction" begin
        include("test_asu.jl")
    end
    
    @testset "ASU Packing" begin
        include("test_packing.jl")
    end
    
    @testset "CFFT Planning" begin
        include("test_cfft.jl")
    end
end
