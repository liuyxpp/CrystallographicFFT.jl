# ASU Calculation Algorithm

This document details the principle and implementation of the automatic Crystallographic Asymmetric Unit (ASU) calculation in `CrystallographicFFT.jl`.

## 1. Core Principle

The algorithm is based on the **Recursive Symmetry Reduction** (or "Parity Filter") approach described by Kudlicki et al. (2007).

### The Idea
For a discrete grid of size $N \times N \times N$, we can classify grid points by their index parity (Even or Odd).
-   **Odd Indices ($2k+1$)**: Generally represent "General Positions" because symmetry operations $x \to -x$ or $x \to Rx+t$ often map ODD points to ODD points in a way that preserves their "bulk-like" nature (or maps them distinctively). The algorithm treats Odd branches as "Leaf Nodes" where no further symmetry reduction is attempted (except for orbit collapsing).
-   **Even Indices ($2k$)**: Represent a coarser sub-grid ($N/2$). These points often lie on special symmetry elements (rotation axes, mirror planes). The algorithm **recurses** into these branches by mapping $2k \to k$.

By recursively filtering "Even" branches, we decompose the grid into a hierarchy of sub-grids (Bulk, Faces, Edges, Corners) with decreasing dimension and increasing symmetry density.

## 2. Implementation Workflow

The implementation in `src/asu.jl` uses a **Breadth-First Search (BFS)** queue to manage the recursion, ensuring it can handle arbitrary dimensions and symmetry groups without hardcoded geometry.

### Step-by-Step Logic

1.  **Initialization**:
    -   Start with the full grid $N$ and the full set of symmetry operations $G$.
    -   Push the initial state to the Queue.

2.  **Processing Loop (BFS)**:
    -   Pop a state $(N_{curr}, \text{Ops}, \text{Offset}, \text{Scale})$.
    -   **Stop Condition**: If all dimensions are marked "General Position" (GP) or $N \le 1$, stop recursion.
        -   **Leaf Processing**: Generate all points in this local sub-grid, compute their full orbits under the current operations, and pick one representative per orbit. Save to ASU.

3.  **Splittability Check (Coupling Detection)**:
    -   Before blindly splitting into Even/Odd, we check if dimensions are **Coupled**.
    -   A dimension $d$ is *Unsplittable* if any symmetry operation maps an Even index to an Odd index (or vice versa) *along that dimension*, or mixes it with another dimension in a way that breaks simpler parity logic.
    -   If a dimension is unsplittable (e.g., due to a glide plane $x \to x+1/2$), we mark it as **GP** immediately and do not split it. This handles non-symmorphic groups like $p2mg$ correctly.

4.  **Branching**:
    -   For valid splittable dimensions, we generate $2^k$ sectors (combinations of Even/Odd).
    -   **Validity Check**: We verify if the current sector is **closed** under the symmetry operations (i.e., operations map the sector to itself). If not (e.g. it maps to a different sector), we discard it (it will be covered by the other sector's orbit). *Note: In the current robust implementation, we process all valid sectors.*
    -   **Recursion**:
        -   **Odd Sector**: Mark as GP (Stop splitting).
        -   **Even Sector**: Reduce $N \to N/2$, update Ops ($R' = S^{-1}RS, t' = \dots$), and push to Queue.

## 3. Data Structures

### `SymOp`
Represents a symmetry operation in integer grid coordinates.
```julia
struct SymOp
    R::Matrix{Int}  # Rotation matrix
    t::Vector{Int}  # Translation vector (in grid units)
end
```

### `ASUPoint`
Represents a final computed independent point.
```julia
struct ASUPoint
    idx::Vector{Int}    # Global coordinate (0-based)
    depth::Vector{Int}  # Recursion depth [kx, ky, kz] (0=Bulk, >0=Special)
    multiplicity::Int   # Size of the orbit (weight for integration)
end
```

## 4. Integration with Crystalline.jl

We typically do not manually input `SymOp`s. Instead, we use `Crystalline.jl`:

```julia
using Crystalline
# Fetch standard operations for Space Group 47 (Pmmm) in 3D
ops = get_ops(47, 3, (32, 32, 32)) 
# get_ops automatically converts fractional translations to integer grid shifts
```

## 5. Verification Methodology

To ensure correctness for *any* space group, we validiate two properties:
1.  **Completeness**: $\bigcup_{p \in \text{ASU}} \text{Orbit}(p) = \text{Grid}$
2.  **Disjointness**: $\text{Orbit}(p_i) \cap \text{Orbit}(p_j) = \emptyset, \forall i \neq j$

This validation is implemented in `test/validate_asu.jl` and has confirmed correctness for $p2mm$, $p2mg$, and $Pmmm$.
