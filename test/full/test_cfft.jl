using Test
using CrystallographicFFT
using CrystallographicFFT.ASU
using FFTW
using LinearAlgebra
using Random

# --- Helpers ---
function expand_to_full_grid(asu::CrystallographicASU{D, T, A}, N::Tuple, ops) where {D, T, A}
    full_grid = zeros(ComplexF64, N)
    for (d, blocks) in asu.dim_blocks
        for block in blocks
            cis = CartesianIndices(size(block.data))
            for index in cis
                val = block.data[index]
                g_idx_0based = [block.range[k][index[k]] for k in 1:D]
                
                orbit = Set{Vector{Int}}([g_idx_0based])
                stack = [g_idx_0based]
                while !isempty(stack)
                    curr = pop!(stack)
                    for op in ops
                        next_p = apply_op(op, curr, N)
                        if !(next_p in orbit)
                            push!(orbit, next_p); push!(stack, next_p)
                        end
                    end
                end
                for p in orbit; full_grid[(p .+ 1)...] = val; end
            end
        end
    end
    return full_grid
end

function verify_spectral_consistency(full_grid, cfft_asu, N)
    D = length(N)
    full_spec = fft(full_grid)
    all_match = true
    
    for (dim_key, blocks) in cfft_asu.dim_blocks
        for block in blocks
            sizes = size(block.data) 
            starts = [first(r) for r in block.range]
            steps = [step(r) for r in block.range]
            
            # Skip partial blocks (simplification)
            if any(sizes .* steps .!= N); continue; end
            
            expected = zeros(ComplexF64, sizes)
            for idx in CartesianIndices(sizes)
                q = Tuple(idx) .- 1 
                sum_val = 0.0 + 0.0im
                r_iterators = [0:(steps[d]-1) for d in 1:D]
                
                for r_tuple in Iterators.product(r_iterators...)
                    r = collect(r_tuple)
                    k = [q[d] + r[d] * sizes[d] for d in 1:D]
                    phase = exp(2π * im * sum(k[d] * starts[d] / N[d] for d in 1:D))
                    k_idx = [mod(k[d], N[d]) + 1 for d in 1:D] 
                    sum_val += full_spec[k_idx...] * phase
                end
                expected[idx] = sum_val / prod(steps)
            end
            
            if norm(block.data - expected) / (norm(expected) + 1e-12) > 1e-10
                all_match = false
            end
        end
    end
    return all_match
end

@testset "CFFT Planning & Execution" begin
    @testset "p2mm (2D)" begin
        sg_num = 6
        N = (16, 16)
        
        # 1. Plan
        plan = plan_cfft(N, sg_num, ComplexF64, Array)
        @test plan isa CFFTPlan
        
        # 2. Random Data
        input_asu = deepcopy(plan.asu)
        for (d, blocks) in input_asu.dim_blocks
            for b in blocks; b.data .= rand(ComplexF64, size(b.data)); end
        end
        
        # 3. Expansion
        _, ops = find_optimal_shift(get_ops(sg_num, 2, N), N)
        full_data = expand_to_full_grid(input_asu, N, ops)
        
        # 4. Forward CFFT
        spectral_asu = deepcopy(plan.asu)
        mul!(spectral_asu, plan, input_asu)
        
        # 5. Verification
        @test verify_spectral_consistency(full_data, spectral_asu, N)
        
        # 6. Inverse CFFT
        recon_asu = deepcopy(plan.asu)
        ldiv!(recon_asu, plan, spectral_asu)
        
        max_err = 0.0
        for (d, blocks) in recon_asu.dim_blocks
            for (i, b) in enumerate(blocks)
                max_err = max(max_err, norm(b.data - input_asu.dim_blocks[d][i].data))
            end
        end
        @test max_err < 1e-10
    end
end
