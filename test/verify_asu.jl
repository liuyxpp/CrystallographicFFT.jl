using CrystallographicFFT
using CrystallographicFFT.ASU: SymOp, calc_asu, ASUPoint

function print_asu(asu, name)
    println("-"^40)
    println("ASU Calculation for $name")
    println("Total Points: $(length(asu))")
    
    # Sort by depth and then index for readability
    sorted_asu = sort(asu, by = p -> (p.depth, p.idx))
    
    # Group by classification (Depth profile)
    # Depth [k_x, k_y]
    # k=0 means Odd branch (GP)
    # k>0 means Even branch (SP)
    
    current_depth = nothing
    for p in sorted_asu
        if p.depth != current_depth
            println("\nDepth $(p.depth):")
            current_depth = p.depth
        end
        println("  Idx: $(p.idx), Orbit Size: $(p.multiplicity)")
    end
    println("-"^40)
end

# --- P2MM Setup ---
# 8x8 Grid
# Ops:
# 1. x, y
# 2. -x, -y
# 3. -x, y
# 4. x, -y
# Note: In modulo arithmetic, -x is (N-x)%N.
# But SymOp uses linear map (R, t). apply_op handles modulo.

p2mm_ops = [
    SymOp([1 0; 0 1], [0, 0]),    # 1
    SymOp([-1 0; 0 -1], [0, 0]),  # 2
    SymOp([-1 0; 0 1], [0, 0]),   # m x
    SymOp([1 0; 0 -1], [0, 0])    # m y
]

asu_p2mm = calc_asu((8, 8), p2mm_ops)
print_asu(asu_p2mm, "p2mm (8x8)")

# Verify counts
# Total expected: 25
@assert length(asu_p2mm) == 25 "p2mm count mismatch! Expected 25, got $(length(asu_p2mm))"

# Check specific counts from doc
# Depth [0,0] (GP, GP) -> Internal -> 4 points (1,1), (1,3), (3,1), (3,3)
# Depth [>0, 0] or [0, >0] -> Faces
# Depth [>0, >0] -> Corners

# --- P2MG Setup ---
# 8x8 Grid
# Ops:
# 1. 1
# 2. 2: (-x, -y)
# 3. m: (-x+4, y) => R=[-1 0; 0 1], t=[4, 0]  (Wait, 4-x = -x+4. Yes)
# 4. g: (x+4, -y) => R=[1 0; 0 -1], t=[4, 0]

p2mg_ops = [
    SymOp([1 0; 0 1], [0, 0]),
    SymOp([-1 0; 0 -1], [0, 0]),
    SymOp([-1 0; 0 1], [4, 0]),
    SymOp([1 0; 0 -1], [4, 0])
]

asu_p2mg = calc_asu((8, 8), p2mg_ops)
print_asu(asu_p2mg, "p2mg (8x8)")

# Verify counts
# Total expected: 21
@assert length(asu_p2mg) == 21 "p2mg count mismatch! Expected 21, got $(length(asu_p2mg))"

# Detailed Check
# x=1 col (GP X): Depth [0, ?].
# Expect X depth 0 points to be full column (8 points).
# x=2 col (Mirror): Depth [1, ?]. (Since 2 is 1st even split of 4, wait. 2 is Even->Odd. Depth 1).
# Expect 8 points.
# x=0 col (Rot): Depth [2, ?]. (0 is Even->Even->Odd(mapped from 0). Depth >=2).
# Expect 5 points.

println("Verification Successful!")
