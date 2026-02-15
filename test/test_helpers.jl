# Shared test helpers — included once by runtests.jl
# Avoids duplicating make_symmetric / fullgrid_reference across test files.

using Random

"""
    make_symmetric(ops, N; seed=42)

Symmetrise random noise under `ops`.  Works for any 3D space group.
"""
function make_symmetric(ops, N; seed=42)
    Random.seed!(seed)
    u = randn(N...)
    u_sym = zeros(N...)
    N1, N2, N3 = N
    y = zeros(Int, 3)
    for op in ops
        R = round.(Int, op.R)
        t = round.(Int, op.t)
        @inbounds for iz in 0:N3-1, iy in 0:N2-1, ix in 0:N1-1
            for d in 1:3
                y[d] = mod(R[d,1]*ix + R[d,2]*iy + R[d,3]*iz + t[d], N[d])
            end
            u_sym[y[1]+1, y[2]+1, y[3]+1] += u[ix+1, iy+1, iz+1]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

"""
    make_symmetric_2d(ops, N; seed=42)

2D version of make_symmetric.
"""
function make_symmetric_2d(ops, N; seed=42)
    Random.seed!(seed)
    u = randn(N...)
    u_sym = zeros(N...)
    for op in ops
        R = round.(Int, op.R)
        t = round.(Int, op.t)
        for iy in 0:N[2]-1, ix in 0:N[1]-1
            x = [ix, iy]
            rx = mod.(R * x .+ t, collect(N))
            u_sym[rx[1]+1, rx[2]+1] += u[ix+1, iy+1]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

"""
Full-grid FFT diffusion reference.  Returns stride-L subgrid of IFFT(K·FFT(u)).
"""
function fullgrid_reference(u_sym, N, Δs, lattice, L)
    F_full = fft(u_sym)
    recip_B = 2π * inv(Matrix(lattice))'
    K_full = zeros(ComplexF64, N...)
    N1, N2, N3 = N
    hc = zeros(3)
    @inbounds for iz in 0:N3-1, iy in 0:N2-1, ix in 0:N1-1
        hc[1] = ix >= N1÷2 ? ix - N1 : ix
        hc[2] = iy >= N2÷2 ? iy - N2 : iy
        hc[3] = iz >= N3÷2 ? iz - N3 : iz
        kv = recip_B * hc
        K_full[ix+1,iy+1,iz+1] = exp(-dot(kv,kv) * Δs)
    end
    f_out = real.(ifft(K_full .* F_full))
    M = N .÷ Tuple(L)
    return Float64[f_out[1+(i-1)*L[1], 1+(j-1)*L[2], 1+(k-1)*L[3]]
                   for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

"""Extract stride-L subgrid."""
function extract_subgrid(u_sym, N, L)
    M = N .÷ Tuple(L)
    return Float64[u_sym[1+(i-1)*L[1], 1+(j-1)*L[2], 1+(k-1)*L[3]]
                   for i in 1:M[1], j in 1:M[2], k in 1:M[3]]
end

"""Compute full orbit of a point under symmetry operations."""
function compute_full_orbit(p::Vector{Int}, ops, N::Tuple)
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
