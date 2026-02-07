using Test
using CrystallographicFFT.ASU
using CrystallographicFFT.SymmetryOps
using Crystalline
using LinearAlgebra

function compute_full_orbit(p::Vector{Int}, ops::Vector{SymOp}, N::Tuple)
    orbit = Set{Vector{Int}}([p])
    stack = [p]
    while !isempty(stack)
        curr = pop!(stack)
        for op in ops
            next_p = apply_op(op, curr, N)
            if !(next_p in orbit)
                push!(orbit, next_p)
                push!(stack, next_p)
            end
        end
    end
    return orbit
end

@testset "ASU Construction & Magic Shift" begin
    
    @testset "Rational Shift Search (p2mm)" begin
        N = (8, 8)
        sg_num = 6 # p2mm
        base_ops = get_ops(sg_num, 2, N)
        
        # Should find half-pixel shift
        shift, shifted_ops = find_optimal_shift(base_ops, N)
        
        @test shift ≈ [0.5/8, 0.5/8]
        
        # Check invariance manually
        valid, _ = check_shift_invariance(base_ops, collect(shift), N)
        @test valid
    end

    @testset "ASU Validation (Coverage & Disjointness)" begin
        # Helper to validate an ASU
        function test_asu_validity(sg_num, dim, N, expected_size)
            ops = get_ops(sg_num, dim, N)
            points, shift = calc_asu(sg_num, dim, N)
            
            # Get consistent shifted ops for validation logic
            _, shifted_ops = find_optimal_shift(ops, N)
            
            # 1. Check Size
            @test length(points) == expected_size
            
            # 2. Check Completeness & Disjointness
            total_points = prod(N)
            coverage = Set{Vector{Int}}()
            
            for p_asu in points
                orb = compute_full_orbit(p_asu.idx, shifted_ops, N)
                
                # Disjointness
                @test isempty(intersect(coverage, orb))
                union!(coverage, orb)
            end
            
            # Completeness
            @test length(coverage) == total_points
        end

        # p2mm (8x8) -> 16 points (All GP)
        test_asu_validity(6, 2, (8, 8), 16)
        
        # p2mg (8x8) -> 16 points
        test_asu_validity(7, 2, (8, 8), 16)
        
        # Pmmm (8x8x8) -> 64 points (512/8)
        test_asu_validity(47, 3, (8, 8, 8), 64)
    end
end
