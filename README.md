# Kalnajs Log Spirals

This project explores normal modes in galactic disks using logarithmic-spiral expansion methods, following the approaches of Kalnajs (1965, 1971) and Zang (1976).

## Overview

The code implements matrix methods for computing non-axisymmetric normal modes in hot stellar disks, focusing on:
- Logarithmic-exponential basis functions for mode decomposition
- Toomre-Zang disk models (n=4 cut-out disks)
- Nonlinear eigenvalue formulations
- Newton-Raphson searches for unstable modes

## Main Files

### Matrix Method Implementations

- **`Matrix_LogExp.m`** - Linear matrix method using logarithmic-exponential basis based on separable kernel approach (Eq. 28). Computes normal modes for Toomre-Zang and Isochrone models.

- **`Matrix_LogExp_ver2.m`** - Alternative version of the matrix method with modified numerical parameters.

- **`Matrix_LogExp_bk1.m`** - Backup/development version of the matrix method.

- **`Solve_Eq16_LogExp.m`** - Solves Eq. 16 using logarithmic-exponential expansion.

### Zang Method (Logarithmic Spirals)

- **`zang_logspiral.m`** - Main implementation of Zang's (1976) matrix method for hot Mestel disks using logarithmic-spiral expansion. Evaluates eigenvalue λ(ω) of the integral equation for m=2, N=4 cut-out disk at Q≈1.

- **`zang_det_scan.m`** - Scans the complex ω-plane to find zeros of det(I - M·dα) where M is the kernel matrix.

- **`zang_newton_search.m`** - Newton-Raphson search for unstable modes in the Zang formulation.

- **`zang_newton_search1.m`**, **`zang_newton_search_2.m`**, **`zang_newton_search_ver2.m`** - Various versions and refinements of the Newton search algorithm.

- **`zang_newton_search_mp.m`** - Multiple-precision version of the Newton search.

- **`zang_newton_search_mp_parallel.m`** - Parallelized multiple-precision Newton search.

- **`zang_newton_search_ver2_test.m`** - Test script for version 2 of the Newton search.

### Nonlinear Eigenvalue Method

- **`NL_precompute.m`** - Precomputes all ω-independent quantities for the nonlinear eigenvalue method (Eqs. 39-46). Creates data structure for subsequent zero-finding.

- **`NL_grid_scan.m`** - Performs grid scan over the complex ω-plane to locate approximate positions of unstable modes.

- **`NL_newton_search.m`** - Newton-Raphson refinement of mode frequencies found by grid scan.

### Utility Functions

- **`gamma_complex.m`** - Complex gamma function implementation for special function evaluations.

- **`show_zeros.m`** - Visualization utility for displaying eigenvalue zeros in the complex plane.

## Usage

Typical workflow:

1. Run `NL_precompute.m` to generate orbit data and ω-independent quantities
2. Run `NL_grid_scan.m` to find approximate mode locations
3. Use `NL_newton_search.m` to refine mode frequencies
4. Alternatively, use `zang_logspiral.m` for direct Zang-method calculations

## Dependencies

The code requires MATLAB function libraries located at:
- `/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/models`
- `/Users/epolyach/icloud/WORK/Astronomy/Normal_modes/Original_Matlab_C`

(Update these paths in individual files as needed for your system)

## References

- Kalnajs, A. J. 1965, PhD thesis, Harvard University
- Kalnajs, A. J. 1971, ApJ, 166, 275
- Zang, T. A. 1976, PhD thesis, MIT
- Toomre, A. 1981, in "Structure and Evolution of Normal Galaxies"

## Authors

E. V. Polyachenko, I. G. Shukhman  
December 2025
