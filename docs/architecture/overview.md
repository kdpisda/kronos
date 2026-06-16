---
title: High-Level Overview
description: KRONOS is a research-grade ab initio plane-wave DFT engine that solves the Kohn-Sham equations self-consistently for periodic crystalline systems.
keywords:
  - KRONOS architecture
  - plane-wave DFT engine
  - ab initio DFT
  - C++ DFT code
  - GPU DFT
  - Kohn-Sham equations
  - pseudopotential DFT
slug: /architecture/overview
sidebar_position: 1
---

# High-Level Overview

KRONOS (Kohn-Residual Optimized Numerics Over Silicon) is a research-grade, ab initio plane-wave pseudopotential DFT engine targeting periodic crystalline systems. It solves the Kohn-Sham equations self-consistently using norm-conserving pseudopotentials, LDA and GGA exchange-correlation functionals, and a GPU abstraction layer for CUDA/HIP offloading. This page describes the outputs, scope, and internal conventions of the engine before diving into the [SCF flowchart](scf-flowchart.md), [component diagram](component-diagram.md), and [algorithms](algorithms.md) in subsequent pages.

KRONOS (Kohn-Residual Optimized Numerics Over Silicon) is a research-grade,
ab initio plane-wave pseudopotential Density Functional Theory (DFT) engine.
It solves the Kohn-Sham equations self-consistently to compute:

- Ground-state total energy and its decomposition (kinetic, Hartree, XC,
  local PP, nonlocal PP, Ewald ion-ion)
- Kohn-Sham eigenvalues and band structure
- Electron density on real-space and reciprocal-space grids
- Hellmann-Feynman ionic forces
- Density of states

KRONOS targets norm-conserving pseudopotentials (NCPP) with LDA and GGA
exchange-correlation functionals. It reads standard UPF v2 pseudopotential
files and YAML input, and writes JSON summaries and HDF5 binary output.

Within the DFT ecosystem, KRONOS occupies the same niche as Quantum ESPRESSO's
PWscf module -- a plane-wave code operating in reciprocal space with periodic
boundary conditions. It is designed for periodic crystalline systems (bulk
solids, surfaces, 2D materials with vacuum padding) using the pseudopotential
approximation to replace core electrons.

All internal quantities use **Rydberg atomic units** (energies in Ry, lengths
in bohr). The code is written in C++20 with a GPU abstraction layer for
future CUDA/HIP offloading.
