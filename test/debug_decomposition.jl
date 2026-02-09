using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.KRFFT: build_recursive_tree, collect_leaves

function analyze_tree(root)
    leaves = collect_leaves(root)
    total_fft = sum(prod(l.subgrid_N) for l in leaves)
    n_leaves = length(leaves)
    max_depth = 0
    function walk(node, d)
        if node.is_leaf
            max_depth = max(max_depth, d)
        else
            for c in node.children
                walk(c, d+1)
            end
        end
    end
    walk(root, 0)
    return (; n_leaves, total_fft, max_depth)
end

function main()
    test_cases = [
        (2,   "P-1    ", 2),
        (47,  "Pmmm   ", 8),
        (62,  "Pnma   ", 8),
        (14,  "P21/c  ", 4),
        (136, "P42/mnm", 16),
        (200, "Pm-3   ", 24),
        (221, "Pm-3m  ", 48),
        (225, "Fm-3m  ", 192),
        (229, "Im-3m  ", 96),
    ]

    for N_val in [8, 16, 32, 64]
        N = (N_val, N_val, N_val)
        vol = prod(N)
        println("\n=== N = $(N_val)^3 = $vol ===")
        println("SG   Name     |G|   leaves  fft_pts   ratio    |G|×    gap")
        println("-"^70)
        for (sg, name, G_order) in test_cases
            ops = get_ops(sg, 3, N)
            root = build_recursive_tree(N, ops)
            info = analyze_tree(root)
            ratio = vol / info.total_fft
            target = Float64(G_order)
            gap = ratio / target
            println("$sg  $name  $(lpad(G_order,3))  $(lpad(info.n_leaves,4))  " *
                    "$(lpad(info.total_fft,7))  $(lpad(round(ratio,digits=1),6))×  " *
                    "$(lpad(round(target,digits=0),4))×  $(round(gap*100,digits=1))%")
        end
    end
end

main()
