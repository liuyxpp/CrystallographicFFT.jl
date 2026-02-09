"""
    debug_tree.jl

Survey recursive tree reduction for all orthorhombic, tetragonal, and cubic space groups.
Shows current reduction ratio, centering index, and predicted ratio with centering.
"""

using CrystallographicFFT.SymmetryOps: get_ops, SymOp
using CrystallographicFFT.KRFFT: build_recursive_tree, collect_leaves
using LinearAlgebra
using Crystalline: label as sg_label, spacegroup

function centering_sublattice_index(cents, N)
    D = length(N)
    if isempty(cents)
        return 1
    end
    group = Set{Vector{Int}}()
    push!(group, zeros(Int, D))
    queue = [zeros(Int, D)]
    while !isempty(queue)
        g = pop!(queue)
        for τ in cents
            h = mod.(g .+ τ, N)
            if h ∉ group
                push!(group, h)
                push!(queue, h)
            end
        end
    end
    return length(group)
end

function centering_label(cent_index)
    cent_index == 1 && return "P"
    cent_index == 2 && return "I/C/A"
    cent_index == 4 && return "F"
    return "?($cent_index)"
end

function analyze_group(sg, D, N)
    ops = get_ops(sg, D, N)
    G_order = length(ops)

    I_mat = Matrix{Float64}(I(D))
    cents = [round.(Int, op.t) for op in ops
             if maximum(abs.(op.R .- I_mat)) < 1e-10 && any(round.(Int, op.t) .!= 0)]
    cent_index = centering_sublattice_index(cents, collect(N))

    root = build_recursive_tree(N, ops)
    leaves = collect_leaves(root)
    total = sum(prod(l.subgrid_N) for l in leaves)
    ratio = prod(N) / total
    ratio_with_cent = ratio * cent_index

    return (; G_order, cent_index, ratio, ratio_with_cent, n_leaves=length(leaves))
end

function main()
    D = 3
    N = (64, 64, 64)  # Large enough for convergence

    # Space group ranges:
    # Orthorhombic: 16-74, Tetragonal: 75-142, Cubic: 195-230
    systems = [
        ("Orthorhombic", 16:74),
        ("Tetragonal",    75:142),
        ("Cubic",        195:230),
    ]

    println("=" ^ 100)
    println("Recursive KRFFT tree survey — N = $(N[1])³")
    println("=" ^ 100)
    println()

    for (system_name, sg_range) in systems
        println("━" ^ 100)
        println("  $system_name (SG $(first(sg_range))–$(last(sg_range)))")
        println("━" ^ 100)
        header = "  SG   Name              |G|  cent  ratio   ×cent  target  efficiency"
        println(header)
        println("  " * "─" ^ (length(header) - 2))

        for sg in sg_range
            # Get HM symbol from Crystalline
            name = try
                string(sg_label(spacegroup(sg, D)))
            catch
                "SG$sg"
            end
            # Truncate/pad name
            name_short = length(name) > 16 ? name[1:16] : rpad(name, 16)

            r = try
                analyze_group(sg, D, N)
            catch e
                println("  $(lpad(sg,3))  $name_short  *** ERROR: $(sprint(showerror, e)) ***")
                continue
            end

            eff = r.ratio_with_cent / r.G_order * 100
            cent_str = centering_label(r.cent_index)

            println("  $(lpad(sg,3))  $name_short " *
                    "$(lpad(r.G_order,4))  $(rpad(cent_str,5)) " *
                    "$(lpad(round(r.ratio, digits=1),6))×  " *
                    "$(lpad(round(r.ratio_with_cent, digits=1),6))×  " *
                    "$(lpad(r.G_order,5))×  " *
                    "$(lpad(round(eff, digits=1),5))%")
        end
        println()
    end

    # Summary: show groups where centering gap exists
    println("=" ^ 100)
    println("Groups with centering gap (cent_index > 1):")
    println("=" ^ 100)
    header = "  SG   Name              |G|  cent  ratio   ×cent  target  gap_factor"
    println(header)
    println("  " * "─" ^ (length(header) - 2))

    for (_, sg_range) in systems
        for sg in sg_range
            r = try
                analyze_group(sg, D, N)
            catch
                continue
            end
            r.cent_index <= 1 && continue

            name = try
                string(sg_label(spacegroup(sg, D)))
            catch
                "SG$sg"
            end
            name_short = length(name) > 16 ? name[1:16] : rpad(name, 16)
            cent_str = centering_label(r.cent_index)
            gap = r.cent_index  # how much FFT we're leaving on the table

            println("  $(lpad(sg,3))  $name_short " *
                    "$(lpad(r.G_order,4))  $(rpad(cent_str,5)) " *
                    "$(lpad(round(r.ratio, digits=1),6))×  " *
                    "$(lpad(round(r.ratio_with_cent, digits=1),6))×  " *
                    "$(lpad(r.G_order,5))×  " *
                    "$(gap)×")
        end
    end
    println()
end

main()
