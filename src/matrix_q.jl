module MatrixQ

using LinearAlgebra
using SparseArrays
using ..SpectralIndexing
using ..ASU
using ..SymmetryOps

export calc_matrix_q, calc_gradient_term

"""
    calc_gradient_term(N::Tuple, Δs::Float64, lattice_vectors::AbstractMatrix) -> Function

Returns a function `kernel(k_idx_vec) -> scalar` that computes the diffusion kernel value.
For standard Gaussian chains: exp(-|k|^2 * Δs).
`lattice_vectors` columns are the primitive lattice vectors (a, b, c).
"""
function calc_gradient_term(N::Tuple, Δs::Float64, lattice_vectors::AbstractMatrix)
    # Reciprocal lattice vectors b_i satisfy a_i . b_j = 2π δ_ij
    # B = 2π * inv(A)' 
    # But usually physics def in SCFT: k = 2π * (n . b_recip_basis?) 
    # Let's assume standard conventions: 
    # Real space r = x/Nx * A[:,1] + ...
    # k vectors are integers n.
    # Physical k = 2π * B * (n ./ N) ? Or just B * n ?
    # Consistent with exp(i k.r). 
    # If r = A * (x_frac), k_phys = B * k_int.
    # r . k_phys = (x_frac)' * A' * B * k_int = x_frac' * 2π * I * k_int = 2π * (x_frac . k_int). Correct.
    
    B = 2π * inv(lattice_vectors)'
    
    return function(k_idx_vec::Vector{Int})
        # k_idx_vec are integers in (-N/2, N/2]
        # Physical vector K = B * (k_idx / N ...) NO.
        # k indices are 0..N-1 usually representing frequency multiples of 2π/L.
        # If box is unit box L=1, then k=2π n.
        # If box is defined by lattice vectors A.
        # The wavevector corresponding to integer index n is K = n_1 * b_1 + n_2 * b_2 + ...
        # where b_i are reciprocal lattice vectors.
        # No division by N here because the integer indices ARE the harmonic numbers.
        
        K = B * k_idx_vec
        k2 = dot(K, K)
        return exp(-k2 * Δs)
    end
end

"""
    calc_matrix_q(indexing::SpectralIndexing, kernel_func::Function) -> AbstractMatrix

Compute the Matrix Q for the diffusion step.
For an isotropic kernel (like exp(-k^2)), Q is diagonal in the symmetry-adapted basis
(which coincides with the orbital basis in this case).

Q_{ii} = kernel_func(k_i)

where k_i is the representative k-vector for the i-th ASU point.

Returns a Diagonal matrix.
"""
function calc_matrix_q(indexing::SpectralIndexing, kernel_func::Function)
    # Since the kernel is isotropic and scaling-invariant (or rather, the kernel function
    # handles the physical metrics), and assuming the kernel commutes with symmetry
    # (true for scalar Laplacian), the Matrix Q is diagonal in the spectral ASU basis.
    
    # Iterate over all spectral ASU points
    n_points = length(indexing.points)
    q_values = Vector{Float64}(undef, n_points)
    
    for (i, p) in enumerate(indexing.points)
        # Get the representative k-vector (integers)
        k_vec = get_k_vector(indexing, i)
        
        # Calculate kernel value
        q_values[i] = kernel_func(k_vec)
    end
    
    return Diagonal(q_values)
end

end
