using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.KRFFT: build_recursive_tree, collect_leaves

function main()
    for (sg, name, G_order) in [
        (225, "Fm-3m", 192),
        (229, "Im-3m", 96),
    ]
        println("=== $name (|G|=$G_order) convergence ===")
        println("N        vol        fft_pts     ratio   /|G|")
        for k in 3:10
            N_val = 2^k
            N = (N_val, N_val, N_val)
            vol = prod(N)
            ops = get_ops(sg, 3, N)
            root = build_recursive_tree(N, ops)
            leaves = collect_leaves(root)
            total = sum(prod(l.subgrid_N) for l in leaves)
            ratio = vol / total
            pct = ratio / G_order * 100
            println("$(lpad(N_val,4))  $(lpad(vol,10))  $(lpad(total,10))  $(round(ratio,digits=2))x  $(round(pct,digits=1))%")
        end
        println()
    end
end

main()
