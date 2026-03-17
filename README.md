# Thermal Conductivity in Nematic Superconductors (Tight-Binding Model)

Numerical calculation of thermal conductivity as a function of temperature from the critical temperature to near 0 in 2D superconducting system using Bogoliubov quasiparticle formalism, based on a tight-binding model with superconductivity emerging from a nematic phase.

---

## Overview

This project implements a computational model to study thermal transport in unconventional supercanductors.

The model combines:

- Tight-binding electronic structure
- Nematic order (treated phenomenologically)
- Superconducting Pairing (Bogoliubov quasiparticle excitations)

Thermal conductivity is computed by integrating quasiparticle contributions to the Green's function over the Fermi surface (FS) deformed by nematic ordering.

---

## Methodology

1. Self-consistent Calculation of Coexisting Order Parameters
     - Nematic order parameter in the absence of superconductivity (SC) is calculated, this is used as an initial "guess" by the numerical solver for the coexisting order parameters. Since the superconducting critical temperautre ($T_c$) is lower than the nematic critical temperature ($T_N$) this functions as the solution above $T_c$
     - Nematicity modifies superconducting order by mixing s- and d-wave ordering, this was treated phenomenologically by setting the degree of mixing parameter (r) from some number between 0 to 1, this is defined by user-input
     - The coexisting superconducting and nematic order parameters are iteratively calculated for all temperatures below $T_c$

2. Calculation of impurity scattering Green's function tensor in Nambu basis along Nematically Deformed FS
     - k-points within energy envelope (from $-E_{cut}$ to $E_{cut}$, where $E_{cut}$ is temperature-dependent to avoid unecessary calculations) in the vicinity of the FS deformed by nematicity in the absence of SC are calculated in order to determine integration-grid
     - Bogoliubov-de Gennes Hamiltonian diagonalized along integration-grid to obtain quasiparticle spectrum
     - Green's function is calculated for every $E$-point within integration grid
     - Green's function is used to determine Bogoliubov quasiparticle lifetimes in both weak (Born approximation) and strong (Unitary approximation) impurity scattering limits 

3. Calculation of Thermal Conductivity Tensor
     - The diagonal terms (off-diagonal terms are zero by symmetry) of the thermal conductivity tensor are calculated for a specific temperature from the Boltzmann kinetic equation from the same integration-grid used to calculate the Green's function previously
     - Diagonal terms are anisotropic due to the symmetry-breaking of the nematic deformation of the electronic state
     - Thermal conductivity tensor components at different temperatures are calculated in parallel on different CPU cores, every step (aside from the first one) is parallelized in this manner

---

## Results

- Captures anisotropic thermal transport behavior characteristic of symmetry-broken nematic superconductors
- Provides a computational framework consistent with phenomenology observed in cuprate SCs
- Demonstrates scalability improvements via parallel execution

---

## Skills Demonstrated

- Tight-binding and Bogoliubov-de Gennes modeling
- Numerical integration in momentum space
- Scientific computing in Python (NumPy, SciPy)
- Parallel computing with multiprocessing (CPU-based)
- Performance optimization of numerical workflows and Python-compilation (Numba)

---

## Notes

This code was developed as part of research into unconventional superconductivity and is intended as a demonstration of numerical modeling techniques rather than a production software package.
