using Crystalline

# Try to load a space group
sg_2mm = spacegroup(25) # Pmm2 is 25? Wait, p2mm is 2D. 
# Crystalline.jl handles 1D, 2D, 3D.
# 2D groups (Plane groups) are 1-17.
# p2mm is number 6? Or p2mm is a different notation.
# Let's check 2D groups.

println("--- 2D Group 6 (p2mm?) ---")
sg2d = spacegroup(6, 2) # 6th subperiodic group in 2D? 
# Or uses sequential numbering?
# Crystalline docs says: `spacegroup(num, dim=3)`
# For 2D, maybe `plane_group`? or just spacegroup(num, 2).

# Let's try to find p2mm and p2mg.
# p2mm corresponds to number 6 in 2D?
# p2mg corresponds to number 7 in 2D?

sg6 = spacegroup(6, 2)
println("Info for SG 6 (2D): ", label(sg6))
ops6 = operations(sg6)
for (i, op) in enumerate(ops6)
    println("Op $i: ", op)
    println("  Rotation: ", op.R)
    println("  Translation: ", op.τ)
end

println("\n--- 2D Group 7 (p2mg?) ---")
sg7 = spacegroup(7, 2)
println("Info for SG 7 (2D): ", label(sg7))
ops7 = operations(sg7)
for (i, op) in enumerate(ops7)
    println("Op $i: ", op)
    println("  Rotation: ", op.R)
    println("  Translation: ", op.τ)
end

println("\n--- 3D Group 47 (Pmmm) ---")
sg47 = spacegroup(47, 3)
println("Info for SG 47: ", label(sg47))
# ops47 = operations(sg47)
# println("Num ops: ", length(ops47))
