using CrystallographicFFT.ASU
using Crystalline

function verify_sg(sg_num, dim, N, expected_count, label_name)
    println("-"^40)
    println("Verifying SG $sg_num ($dim-D) - $label_name")
    
    ops = get_ops(sg_num, dim, N)
    println("Loaded $(length(ops)) operations.")
    
    asu = calc_asu(sg_num, dim, N)
    println("ASU Size: $(length(asu))")
    
    if length(asu) == expected_count
        println("✅ Count Match!")
    else
        println("❌ Count Mismatch! Expected $expected_count")
    end
end

# Verify p2mm (SG #6 in 2D) on 8x8
verify_sg(6, 2, (8, 8), 25, "p2mm")

# Verify p2mg (SG #7 in 2D) on 8x8
verify_sg(7, 2, (8, 8), 21, "p2mg")

# Verify Pmmm (SG #47 in 3D) on 8x8x8?
# Pmmm should be 1/8th of volume + boundary terms.
# 8x8x8 = 512. 
# Let's try to verify against manual calculation or just run it to see if it works.
verify_sg(47, 3, (8, 8, 8), 1, "Pmmm")
