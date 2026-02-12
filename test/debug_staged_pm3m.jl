## Debug: Verify KRFFT V symmetry relations for Pm-3m (SG 221)
#
# Step 0 of KRFFT V2 Phase 0.
# Verifies the mathematical identities from KRFFT V §5 and Appendix A
# that underpin the A8(1) + P3c + yx staged decomposition.
#
# Run: julia --project=test test/debug_staged_pm3m.jl

using FFTW
using LinearAlgebra
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops, SymOp, apply_op
using CrystallographicFFT.ASU: find_optimal_shift

# ─── Helpers ──────────────────────────────────────────────────────────────

"""Make Pm-3m symmetric data on (2N)³ grid."""
function make_symmetric_pm3m(ops, N_full)
    u = rand(N_full...)
    s = zeros(N_full...)
    for op in ops
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(op.R * x .+ round.(Int, op.t), collect(N_full)) .+ 1
            s[idx] += u[x2...]
        end
    end
    s ./= length(ops)
    return s
end

"""Extract stride-2 subgrid f_{n,m,l}(x₁) from full grid u(x)."""
function extract_subgrid(u, n::Int, m::Int, l::Int, N_half)
    # u is (2N)³, extract points at (2x₁+n, 2x₂+m, 2x₃+l) for x₁ ∈ 0:N-1
    sub = zeros(ComplexF64, N_half...)
    for k in 0:N_half[3]-1, j in 0:N_half[2]-1, i in 0:N_half[1]-1
        # 1-based indexing in Julia: u[2i+n+1, 2j+m+1, 2k+l+1]
        ii = mod(2i + n, 2N_half[1]) + 1
        jj = mod(2j + m, 2N_half[2]) + 1
        kk = mod(2k + l, 2N_half[3]) + 1
        sub[i+1, j+1, k+1] = u[ii, jj, kk]
    end
    return sub
end

"""
Compute twiddle factor: e_A(h₁, t_g) = exp(-2πi h₁·t_g / (2N))
where h₁ is the subgrid frequency and t_g is the translation (in grid coords).
"""
function twiddle(h1, t_g, N_full)
    phase = sum(h1 .* t_g ./ N_full)
    return cispi(-2 * phase)
end

# ─── Test Parameters ──────────────────────────────────────────────────────

println("=" ^ 70)
println("KRFFT V Symmetry Verification for Pm-3m (SG 221)")
println("=" ^ 70)

for N_half_val in [4, 8]
    N_half = (N_half_val, N_half_val, N_half_val)
    N_full = 2 .* N_half
    println("\n" * "─" ^ 70)
    println("N_half = $N_half, N_full = $N_full")
    println("─" ^ 70)

    # Get shifted ops
    ops = get_ops(221, 3, N_full)
    shift, ops_s = find_optimal_shift(ops, N_full)
    println("Shift: $shift")
    println("Number of ops: $(length(ops_s))")
    
    # Generate symmetric data
    u = make_symmetric_pm3m(ops_s, N_full)
    
    # Verify symmetry
    sym_err = 0.0
    for op in ops_s
        for idx in CartesianIndices(u)
            x = collect(Tuple(idx)) .- 1
            x2 = mod.(op.R * x .+ round.(Int, op.t), collect(N_full))
            err = abs(u[idx] - u[(x2 .+ 1)...])
            sym_err = max(sym_err, err)
        end
    end
    println("Symmetry verification: max error = $sym_err")

    # ═══ Section 1: A8(1) Verification ═══════════════════════════════════
    println("\n=== A8(1): Stride-2 Subgrid Relations ===")
    
    # Extract all 8 subgrids and FFT them
    F_sub = Dict{Tuple{Int,Int,Int}, Array{ComplexF64,3}}()
    for n in 0:1, m in 0:1, l in 0:1
        f = extract_subgrid(u, n, m, l, N_half)
        F_sub[(n,m,l)] = fft(f)
    end
    
    # Identify key symmetry operations by their R matrix
    # β: (x,y,z) → (-x,-y,z), i.e. R=diag(-1,-1,1)
    # γ: (x,y,z) → (-x,y,-z), i.e. R=diag(-1,1,-1)
    # ν (inversion): R=diag(-1,-1,-1)
    # α (C₃): R = [0 0 1; 1 0 0; 0 1 0] (x→y→z→x cycle)
    # σ (yx mirror): R = [0 1 0; 1 0 0; 0 0 1] (swap x,y)
    
    function find_op_by_R(ops, R_target)
        for op in ops
            if op.R == R_target
                return op
            end
        end
        return nothing
    end
    
    R_beta  = [-1 0 0; 0 -1 0; 0 0 1]
    R_gamma = [-1 0 0; 0 1 0; 0 0 -1]
    R_nu    = [-1 0 0; 0 -1 0; 0 0 -1]
    R_alpha = [0 0 1; 1 0 0; 0 1 0]
    R_sigma = [0 1 0; 1 0 0; 0 0 1]
    
    op_beta  = find_op_by_R(ops_s, R_beta)
    op_gamma = find_op_by_R(ops_s, R_gamma)
    op_nu    = find_op_by_R(ops_s, R_nu)
    op_alpha = find_op_by_R(ops_s, R_alpha)
    op_sigma = find_op_by_R(ops_s, R_sigma)
    
    println("  β  found: $(op_beta !== nothing), t=$(op_beta===nothing ? "N/A" : op_beta.t)")
    println("  γ  found: $(op_gamma !== nothing), t=$(op_gamma===nothing ? "N/A" : op_gamma.t)")
    println("  ν  found: $(op_nu !== nothing), t=$(op_nu===nothing ? "N/A" : op_nu.t)")
    println("  α  found: $(op_alpha !== nothing), t=$(op_alpha===nothing ? "N/A" : op_alpha.t)")
    println("  σ  found: $(op_sigma !== nothing), t=$(op_sigma===nothing ? "N/A" : op_sigma.t)")
    
    # Verify: F_{nml}(h₁) relates to F_{000}(R_g^T h₁) via phase factors
    # KRFFT V eq 633 (generalized):
    # For β: F_{110}(h₁) = e_A(h₁, t_β) · F_{000}(R_β^T h₁)
    # Because β maps subgrid (0,0,0) to subgrid (1,1,0) (β flips x,y; keeps z)
    
    # Let's verify the general relation:
    # f_{nml}(x₁) = f(2x₁ + (n,m,l)) 
    # Under symmetry g: f(gx) = f(x), so f(x) = f(R_g x + t_g)
    # This means f_{nml} relates to f_{n'ml'} where (n',m',l') = R_g (n,m,l) + t_g mod 2
    
    # For each symmetry operation g, find the subgrid mapping:
    # (n',m',l') = R_g · (n,m,l) + t_g mod 2
    # F_{n'ml'}(h₁) = e_A(h₁, ψ_g) · F_{nml}(R_g^T h₁)
    # where ψ_g is derived from g
    
    println("\n  Subgrid mapping under symmetry operations:")
    
    for (name, op) in [("β", op_beta), ("γ", op_gamma), ("ν", op_nu), ("α", op_alpha), ("σ", op_sigma)]
        if op === nothing; continue; end
        t_mod2 = mod.(round.(Int, op.t), 2)
        println("  $name: t mod 2 = $t_mod2")
        for n in 0:1, m in 0:1, l in 0:1
            nml = [n, m, l]
            nml2 = mod.(op.R * nml .+ t_mod2, 2)
            if nml != nml2
                println("    ($n,$m,$l) → ($(nml2[1]),$(nml2[2]),$(nml2[3]))")
            end
        end
    end
    
    # ═══ Verify: F₁₁₀ = tw · F₀₀₀(R_β^T h) ═════════════════════════════
    println("\n  --- Verifying A8 subgrid equivalences ---")
    
    # General verification: for each pair of subgrids related by a symmetry op,
    # check that F_{target}(h₁) = phase(h₁) · F_{source}(R^T h₁)
    function verify_subgrid_relation(F_sub, nml_src, nml_tgt, op, N_half, N_full)
        F_src = F_sub[Tuple(nml_src)]
        F_tgt = F_sub[Tuple(nml_tgt)]
        R = op.R
        t = round.(Int, op.t)
        
        max_err = 0.0
        for k in 0:N_half[3]-1, j in 0:N_half[2]-1, i in 0:N_half[1]-1
            h1 = [i, j, k]
            # Target frequency
            val_tgt = F_tgt[i+1, j+1, k+1]
            
            # Source: R^T h₁ mod N_half
            h1_rot = mod.(R' * h1, collect(N_half))
            val_src = F_src[h1_rot[1]+1, h1_rot[2]+1, h1_rot[3]+1]
            
            # Phase: exp(-2πi h₁ · t / N_full) where t is the full grid translation
            # But we need to be more careful: the relationship between subgrids
            # involves a twiddle factor that depends on (n,m,l) and (n',m',l')
            #
            # The general formula (KRFFT V Appendix A eq 27):
            # F(h₀,h₁) = Σ_g e_A(A₁ᵀ h₀, t_g) · e_A(h₁, t_g) · Y(R_g^T h₁)
            #
            # For stride-2: A₁ = diag(2,2,2), h₀ = (n,m,l)/2 effectively
            # The twiddle for subgrid mapping is:
            # e_A(h₁, t_g)  where t_g is the grid-coordinate translation
            
            # Simpler: from the DFT definition,
            # F_{nml}(h₁) = Σ_{x₁} f(2x₁+(n,m,l)) · exp(-2πi h₁·x₁/N_half)
            # If f(x) = f(R·x + t), then:
            # F_{nml}(h₁) = exp(-2πi h₁·t_sub/(N_half)) · F_{n'm'l'}(R^T h₁)
            # where t_sub = floor((R·(n,m,l) + t - (n',m',l'))/2)
            
            nml_rot = op.R * collect(nml_src) .+ t
            nml_rot_mod2 = mod.(nml_rot, 2)
            t_sub = div.(nml_rot .- nml_rot_mod2, 2)
            
            phase = cispi(-2 * sum(h1 .* t_sub ./ collect(N_half)))
            val_predicted = phase * val_src
            
            err = abs(val_tgt - val_predicted)
            max_err = max(max_err, err)
        end
        return max_err
    end
    
    # Test all symmetry-related subgrid pairs
    for (name, op) in [("β", op_beta), ("γ", op_gamma), ("ν", op_nu)]
        if op === nothing; continue; end
        t_mod2 = mod.(round.(Int, op.t), 2)
        for n in 0:1, m in 0:1, l in 0:1
            nml_src = [n, m, l]
            nml_tgt = mod.(op.R * nml_src .+ t_mod2, 2)
            if nml_src == nml_tgt; continue; end  # SP, skip
            
            err = verify_subgrid_relation(F_sub, nml_src, nml_tgt, op, N_half, N_full)
            status = err < 1e-10 ? "✓" : "✗"
            println("  $status $name: F_$(Tuple(nml_src)) → F_$(Tuple(nml_tgt)), max err = $(round(err, sigdigits=3))")
        end
    end
    
    # ═══ Verify: A8 butterfly reconstruction ══════════════════════════════
    println("\n  --- Verifying A8 butterfly reconstruction ---")
    
    # Full DFT: F(h) = Σ_{n,m,l} tw_{nml}(h) · F_{nml}(h₁)
    # where h = h₁ + N_half · h₀, h₀ = (n,m,l) ∈ {0,1}³
    # and tw_{nml}(h) = exp(-2πi (n·₁+m·h₂+l·h₃) / (2N))
    # Wait, more precisely:
    # F(h) = Σ_{(n,m,l)} exp(-2πi h·(n,m,l)/(2,2,2)) · F_{nml}(h₁)
    # where h₁ = h mod N_half
    
    F_full_ref = fft(complex(u))
    
    max_recon_err = 0.0
    n_tested = 0
    for kz in 0:N_full[3]-1, ky in 0:N_full[2]-1, kx in 0:N_full[1]-1
        h = [kx, ky, kz]
        h1 = mod.(h, collect(N_half))
        h0 = div.(h, collect(N_half))  # (0 or 1)³
        
        val = zero(ComplexF64)
        for l in 0:1, m in 0:1, n in 0:1
            # Twiddle: exp(-2πi (n·hx + m·hy + l·hz) / (2N))
            # = exp(-2πi Σ (n,m,l)_d · h_d / N_full_d)
            tw = cispi(-2 * sum([n,m,l] .* h ./ collect(N_full)))
            val += tw * F_sub[(n,m,l)][h1[1]+1, h1[2]+1, h1[3]+1]
        end
        
        ref = F_full_ref[kx+1, ky+1, kz+1]
        err = abs(val - ref)
        max_recon_err = max(max_recon_err, err)
        n_tested += 1
    end
    status = max_recon_err < 1e-10 ? "✓" : "✗"
    println("  $status A8 butterfly reconstruction: max err = $(round(max_recon_err, sigdigits=3)) ($n_tested points)")
    
    # ═══ Verify: Only F₀₀₀ needed (via symmetry) ═════════════════════════
    println("\n  --- Verifying: all 8 subgrids derivable from F₀₀₀ ---")
    
    # For Pm-3m with b=1/2 shift:
    # The 8 subgrids {nml} form orbits under {β, γ, βγ, ν}
    # We need to check: can all 7 non-identity subgrids be expressed
    # in terms of F₀₀₀?
    
    # Map: for each (n,m,l) ≠ (0,0,0), find which op maps (0,0,0) → (n,m,l)
    println("  Subgrid mapping from (0,0,0):")
    for (name, op) in [("β", op_beta), ("γ", op_gamma), ("ν", op_nu),
                         ("βγ", op_beta === nothing || op_gamma === nothing ? nothing :
                          SymOp(op_beta.R * op_gamma.R, 
                                op_beta.R * op_gamma.t .+ op_beta.t))]
        if op === nothing; continue; end
        t_mod2 = mod.(round.(Int, op.t), 2)
        nml_tgt = mod.(op.R * [0,0,0] .+ t_mod2, 2)
        println("    $name: (0,0,0) → $(Tuple(nml_tgt))")
    end
    
    # ═══ Section 2: P3c Verification ══════════════════════════════════════
    println("\n=== P3c: C₃ Diagonal Rotation ===")
    
    # F₀₀₀ has C₃ symmetry: F₀₀₀(h) = F₀₀₀(R_α^T h)
    # where α: (x,y,z) → (y,z,x), R_α^T: (h,k,l) → (k,l,h)
    
    F000 = F_sub[(0,0,0)]
    
    # Verify C₃ symmetry of F₀₀₀
    max_c3_err = 0.0
    for k in 0:N_half[3]-1, j in 0:N_half[2]-1, i in 0:N_half[1]-1
        h = [i, j, k]
        # R_α^T h = (j, k, i)
        h_rot = mod.([j, k, i], collect(N_half))
        err = abs(F000[i+1, j+1, k+1] - F000[h_rot[1]+1, h_rot[2]+1, h_rot[3]+1])
        max_c3_err = max(max_c3_err, err)
    end
    status = max_c3_err < 1e-10 ? "✓" : "✗"
    println("  $status F₀₀₀ has C₃ symmetry: max err = $(round(max_c3_err, sigdigits=3))")
    
    # Verify yx mirror symmetry of F₀₀₀
    max_yx_err = 0.0
    for k in 0:N_half[3]-1, j in 0:N_half[2]-1, i in 0:N_half[1]-1
        h = [i, j, k]
        h_mirror = mod.([j, i, k], collect(N_half))
        err = abs(F000[i+1, j+1, k+1] - F000[h_mirror[1]+1, h_mirror[2]+1, h_mirror[3]+1])
        max_yx_err = max(max_yx_err, err)
    end
    status = max_yx_err < 1e-10 ? "✓" : "✗"
    println("  $status F₀₀₀ has yx mirror symmetry: max err = $(round(max_yx_err, sigdigits=3))")
    
    # ═══ P3c: stride-2 sub-subgrids of F₀₀₀ ══════════════════════════════
    println("\n  --- P3c stride-2 sub-subgrids of F₀₀₀ ---")
    
    # f₀₀₀ is the real-space data on the (0,0,0) subgrid (N³ points)
    f000 = extract_subgrid(u, 0, 0, 0, N_half)
    
    # Now do stride-2 split of f₀₀₀ into 8 sub-subgrids
    N_quarter = N_half .÷ 2
    if all(N_quarter .>= 1)
        # Extract sub-subgrids of f₀₀₀
        F_sub2 = Dict{Tuple{Int,Int,Int}, Array{ComplexF64,3}}()
        for n in 0:1, m in 0:1, l in 0:1
            sub = zeros(ComplexF64, N_quarter...)
            for k in 0:N_quarter[3]-1, j in 0:N_quarter[2]-1, i in 0:N_quarter[1]-1
                ii = mod(2i + n, N_half[1]) + 1
                jj = mod(2j + m, N_half[2]) + 1
                kk = mod(2k + l, N_half[3]) + 1
                sub[i+1, j+1, k+1] = f000[ii, jj, kk]
            end
            F_sub2[(n,m,l)] = fft(sub)
        end
        
        # Verify C₃ orbit: F₂_{100} ↔ F₂_{001} ↔ F₂_{010}
        # α: (x,y,z) → (y,z,x), so (1,0,0) → (0,0,1), (0,0,1) → (0,1,0)
        # F₂_{100}(h) should equal F₂_{001}(R_α^T h) (up to twiddle)
        
        # For the sub-subgrids, the C₃ acts as a permutation of the parity indices
        # Since f₀₀₀ has C₃ symmetry, f₀₀₀(x,y,z) = f₀₀₀(y,z,x)
        # Sub-subgrid extraction at (n,m,l):
        # f₂_{nml}(x₂) = f₀₀₀(2x₂ + (n,m,l))
        # Under C₃: f₀₀₀(2x₂+(n,m,l)) = f₀₀₀(2x₂'+(m,l,n))
        # where x₂' = (x₂_y, x₂_z, x₂_x)
        # So f₂_{nml}(x₂) = f₂_{mln}(α·x₂)
        # → F₂_{nml}(h) = F₂_{mln}(R_α^T h)  (no twiddle, since C₃ is pure rotation with no translation for sub-subgrids)
        
        println("  C₃ orbit verification on P3c sub-subgrids:")
        for (nml_src, nml_tgt) in [([1,0,0], [0,0,1]), ([0,0,1], [0,1,0]), ([0,1,0], [1,0,0]),
                                     ([1,1,0], [1,0,1]), ([1,0,1], [0,1,1]), ([0,1,1], [1,1,0])]
            max_err = 0.0
            for k in 0:N_quarter[3]-1, j in 0:N_quarter[2]-1, i in 0:N_quarter[1]-1
                h = [i, j, k]
                # R_α^T h = (j, k, i) — cycle h,k,l
                h_rot = mod.([j, k, i], collect(N_quarter))
                val_src = F_sub2[Tuple(nml_src)][i+1, j+1, k+1]
                val_tgt = F_sub2[Tuple(nml_tgt)][h_rot[1]+1, h_rot[2]+1, h_rot[3]+1]
                err = abs(val_src - val_tgt)
                max_err = max(max_err, err)
            end
            status = max_err < 1e-10 ? "✓" : "✗"
            println("    $status F₂_$(Tuple(nml_src))(h) = F₂_$(Tuple(nml_tgt))(R_α^T h): max err = $(round(max_err, sigdigits=3))")
        end
        
        # Verify SP sub-subgrids: F₂_{000} and F₂_{111} are C₃-invariant
        for sp_nml in [(0,0,0), (1,1,1)]
            F_sp = F_sub2[sp_nml]
            max_err = 0.0
            for k in 0:N_quarter[3]-1, j in 0:N_quarter[2]-1, i in 0:N_quarter[1]-1
                h_rot = mod.([j, k, i], collect(N_quarter))
                err = abs(F_sp[i+1, j+1, k+1] - F_sp[h_rot[1]+1, h_rot[2]+1, h_rot[3]+1])
                max_err = max(max_err, err)
            end
            status = max_err < 1e-10 ? "✓" : "✗"
            println("    $status F₂_$sp_nml is C₃-invariant: max err = $(round(max_err, sigdigits=3))")
        end
        
        # Verify: GP sub-subgrids have yx mirror symmetry
        println("\n  yx mirror symmetry of P3c sub-subgrids:")
        for nml in [(0,0,1), (1,1,0)]
            F_gp = F_sub2[Tuple(nml)]
            max_err = 0.0
            for k in 0:N_quarter[3]-1, j in 0:N_quarter[2]-1, i in 0:N_quarter[1]-1
                h_mirror = mod.([j, i, k], collect(N_quarter))
                err = abs(F_gp[i+1, j+1, k+1] - F_gp[h_mirror[1]+1, h_mirror[2]+1, h_mirror[3]+1])
                max_err = max(max_err, err)
            end
            status = max_err < 1e-10 ? "✓" : "✗"
            println("    $status F₂_$(Tuple(nml)) has yx symmetry: max err = $(round(max_err, sigdigits=3))")
        end
        
        # Verify: SP sub-subgrids also have yx mirror symmetry
        for sp_nml in [(0,0,0), (1,1,1)]
            F_sp = F_sub2[sp_nml]
            max_err = 0.0
            for k in 0:N_quarter[3]-1, j in 0:N_quarter[2]-1, i in 0:N_quarter[1]-1
                h_mirror = mod.([j, i, k], collect(N_quarter))
                err = abs(F_sp[i+1, j+1, k+1] - F_sp[h_mirror[1]+1, h_mirror[2]+1, h_mirror[3]+1])
                max_err = max(max_err, err)
            end
            status = max_err < 1e-10 ? "✓" : "✗"
            println("    $status F₂_$sp_nml has yx symmetry: max err = $(round(max_err, sigdigits=3))")
        end
    end
    
    # ═══ P3c butterfly reconstruction verification ════════════════════════
    println("\n  --- P3c butterfly reconstruction ---")
    
    # F₀₀₀(h₁) = Σ_{(n,m,l)} tw_{nml}(h₁) · F₂_{nml}(h₂)
    # where h₁ = h₂ + N_quarter · h₀, h₀ ∈ {0,1}³
    
    if all(N_quarter .>= 1)
        max_p3c_recon_err = 0.0
        for kz in 0:N_half[1]-1, ky in 0:N_half[2]-1, kx in 0:N_half[3]-1
            h1 = [kx, ky, kz]
            h2 = mod.(h1, collect(N_quarter))
            
            val = zero(ComplexF64)
            for l in 0:1, m in 0:1, n in 0:1
                tw = cispi(-2 * sum([n,m,l] .* h1 ./ collect(N_half)))
                val += tw * F_sub2[(n,m,l)][h2[1]+1, h2[2]+1, h2[3]+1]
            end
            
            ref = F000[kx+1, ky+1, kz+1]
            err = abs(val - ref)
            max_p3c_recon_err = max(max_p3c_recon_err, err)
        end
        status = max_p3c_recon_err < 1e-10 ? "✓" : "✗"
        println("  $status P3c butterfly reconstruction: max err = $(round(max_p3c_recon_err, sigdigits=3))")
    end
    
    # ═══ Section 3: yx Verification ═══════════════════════════════════════
    println("\n=== yx: Diagonal Mirror ===")
    
    # For GP sub-subgrid F₂_{001}(h), verify that stride-2 in xy gives
    # subgrids with mirror symmetry
    if all(N_quarter .>= 2)
        F_gp = F_sub2[(0,0,1)]
        f_gp = ifft(F_gp)  # back to real space
        N_eighth = N_quarter .÷ 2
        
        # stride-2 split in x,y only → 4 sub-sub-subgrids
        F_sub3 = Dict{Tuple{Int,Int}, Array{ComplexF64,3}}()
        for n in 0:1, m in 0:1
            sub = zeros(ComplexF64, N_eighth[1], N_eighth[2], N_quarter[3])
            for k in 0:N_quarter[3]-1, j in 0:N_eighth[2]-1, i in 0:N_eighth[1]-1
                ii = mod(2i + n, N_quarter[1]) + 1
                jj = mod(2j + m, N_quarter[2]) + 1
                sub[i+1, j+1, k+1] = f_gp[ii, jj, k+1]
            end
            F_sub3[(n,m)] = fft(sub)
        end
        
        # yx mirror on parity indices: (n,m) → (m,n)
        # So (0,1) ↔ (1,0): this is a GP pair
        # (0,0) and (1,1): related by mirror (SP pair)
        println("  yx mirror relation on sub-sub-subgrids:")
        
        # Verify (0,1) ↔ (1,0)
        max_err = 0.0
        for k in 0:N_quarter[3]-1, j in 0:N_eighth[2]-1, i in 0:N_eighth[1]-1
            h = [i, j, k]
            h_mirror = [mod(j, N_eighth[1]), mod(i, N_eighth[2]), k]
            val_01 = F_sub3[(0,1)][i+1, j+1, k+1]
            val_10 = F_sub3[(1,0)][h_mirror[1]+1, h_mirror[2]+1, k+1]
            err = abs(val_01 - val_10)
            max_err = max(max_err, err)
        end
        status = max_err < 1e-10 ? "✓" : "✗"
        println("    $status F₃_{01}(h,k,l) = F₃_{10}(k,h,l): max err = $(round(max_err, sigdigits=3))")
    end
    
    # ═══ Summary ══════════════════════════════════════════════════════════
    println("\n=== Operation Count Summary ===")
    n_total = prod(N_full)
    n_asu = n_total ÷ 48
    println("  Full grid: $(N_full) = $n_total points")
    println("  |G| = 48")
    println("  ASU size: $n_asu points")
    println("  A8 reduction: $(n_total) → $(prod(N_half)) (8×)")
    if all(N_quarter .>= 1)
        n_gp = 2 * prod(N_quarter)
        n_sp = 2 * prod(N_quarter)
        println("  P3c level 0: $(prod(N_half)) → $n_gp GP + $n_sp SP = $(n_gp + n_sp)")
        println("    GP: 2 independent FFTs of size $(N_quarter)")  
        println("    SP: 2 grids for recursion (retain C₃+yx)")
    end
end

println("\n" * "=" ^ 70)
println("Verification complete.")
println("=" ^ 70)
