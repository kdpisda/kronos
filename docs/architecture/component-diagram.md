---
title: Component Diagram
description: Module dependency graph for KRONOS — how io/, core/, basis/, potential/, hamiltonian/, solver/, and gpu/ relate to each other.
keywords:
  - DFT software architecture
  - plane-wave DFT modules
  - KRONOS components
  - C++ module dependencies
  - DFT solver architecture
  - hamiltonian module
  - gpu abstraction layer
slug: /architecture/component-diagram
sidebar_position: 3
---

# Component Diagram

KRONOS is organized as a layered set of modules under `src/`, each with strict dependency rules that prevent circular imports and keep the GPU abstraction boundary clean. The diagram below shows how data and control flow from the entry point down through the physics layers. See [Source Layout](source-layout.md) for the full file-by-file breakdown, and [GPU Portability](gpu-portability.md) for how the `gpu/` abstraction layer is structured.

Module dependencies flow top-to-bottom. Each box is a directory under `src/`.

```mermaid
flowchart TD
    IO["io/<br/>YAML in · UPF load · JSON out"]
    CORE["core/<br/>Crystal · Types · Consts"]
    BASIS["basis/<br/>PlaneWaveBasis · FFTGrid · KPoints"]
    POT["potential/<br/>Hartree · XC · LocalPP · NonlocalPP<br/>Ewald · Forces · GGA Gradient"]
    HAM["hamiltonian/<br/>H ψ apply (T + V_loc + V_NL)"]
    SOLVER["solver/<br/>Davidson · DIIS · Fermi · SCF · BFGS"]
    POST["postprocessing/<br/>BandStructure · DOS"]
    GPU["gpu/<br/>FFT · BLAS · Memory<br/>(stubs / CUDA / HIP / Metal)"]
    UTILS["utils/<br/>Timer · Logger · MPI wrapper"]
    MAIN["main.cpp<br/>Entry point"]

    IO --> CORE
    IO --> BASIS
    CORE --> POT
    BASIS --> POT
    POT --> HAM
    HAM --> SOLVER
    SOLVER --> POST
    HAM -. uses .-> GPU
    SOLVER -. uses .-> GPU
    UTILS -. cross-cutting .-> SOLVER
    UTILS -. cross-cutting .-> HAM
    MAIN --> IO
    MAIN --> SOLVER

    classDef hot fill:#fce4a8,stroke:#a06000,color:#000
    class HAM,SOLVER hot
```

Dependency rules:
- `core/` depends on nothing (leaf module).
- `basis/` depends on `core/`.
- `io/` depends on `core/` (Crystal, types).
- `potential/` depends on `core/`, `basis/`, `io/` (for `PseudoPotential` data).
- `hamiltonian/` depends on `basis/`, `potential/` (specifically `NonlocalPP`).
- `solver/` depends on everything above; it orchestrates the full calculation.
- `gpu/` is called by `hamiltonian/` and `basis/` but physics code never
  calls vendor APIs directly -- only the `gpu::` namespace.
- `utils/` is available to all modules (timer, logger).
