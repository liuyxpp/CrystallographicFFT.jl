
using LinearAlgebra

function check_grid_invariance(N, delta, sym_op_func)
    # Define Grid Points: x_n = n/N + delta
    # Symmetry Op: x' = sym_op_func(x)
    # Check if x' corresponds to some m/N + delta (modulo 1)
    
    println("Checking Invariance for delta = $delta with N=$N")
    
    for n in 0:N-1
        x = n/N + delta
        x_prime = sym_op_func(x)
        
        # We need x_prime = m/N + delta + k (integer k)
        # So m = N * (x_prime - delta)
        # m must be integer (close to integer)
        
        m_float = N * (x_prime - delta)
        m_rounded = round(m_float)
        diff = abs(m_float - m_rounded)
        
        if diff > 1e-5
            println("  [FAIL] Point n=$n (x=$x) maps to x'=$x_prime")
            println("         Required m=$m_float is NOT integer (diff=$diff)")
            return false
        end
    end
    println("  [PASS] Grid is invariant.")
    return true
end

# p2mm symmetry: x -> -x (mirror at x=0)
op_mirror(x) = -x

println("--- Test 1: No Shift (delta=0) ---")
check_grid_invariance(8, 0.0, op_mirror)

println("\n--- Test 2: Half-Integer Shift (delta=0.5/8) ---")
# Note: delta is in unit of 1, so 0.5 grid spacing is 0.5/N
check_grid_invariance(8, 0.5/8, op_mirror)

println("\n--- Test 3: Irrational Shift (delta = sqrt(2)/100) ---")
check_grid_invariance(8, sqrt(2)/100, op_mirror)
