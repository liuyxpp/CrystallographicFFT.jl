# ============================================================================
# KernelAbstractions @kernel functions for Device-Agnostic CFFT
# ============================================================================
#
# Phase 1-2: CPU-only implementations using direct loops in execute.jl.
# KA @kernel versions will be added once correctness is verified.
# This file is reserved for future KA kernel implementations.
# ============================================================================

using KernelAbstractions

# ── Phase 1: General Forward Kernels ─────────────────────────────────────────
# Currently handled by _reconstruct_pmmm! and _reconstruct_general! in execute.jl.
# KA @kernel versions pending.

# ── Phase 2: General Backward Kernels ────────────────────────────────────────
# Currently handled by _inv_reconstruct! in execute.jl.

# ── Phase 3: Centered Path Kernels (future) ──────────────────────────────────
# ── Phase 4: Grid Conversion Kernels (future) ────────────────────────────────
