---
title: Observability and Error Handling
description: KRONOS structured JSON logging, scoped profiling with KRONOS_TIMER, SCF step output format, and the full error handling table with exit codes.
keywords:
  - DFT logging
  - NVTX profiling
  - SCF diagnostics
  - structured logs
  - JSON logging DFT
  - KRONOS error handling
  - SCF convergence errors
  - DFT profiling
slug: /architecture/observability
sidebar_position: 9
---

# Observability and Error Handling

KRONOS emits structured JSON logs to stderr for machine-readable monitoring, prints human-readable SCF step progress to stdout, and records wall-time profiling data via the `KRONOS_TIMER` RAII macro. On failure, errors are categorized by exit code so that automation scripts can distinguish convergence failure from input errors from pseudopotential parse errors. This page documents the log format, the profiling API, the SCF step output format, and the full error condition table.

### Structured Logging

The `Logger` singleton emits JSON-formatted lines to stderr:

```json
{"timestamp":"2026-03-06T12:34:56Z","level":"info","event":"scf","message":"SCF solver initialized","num_bands":"8","num_pw":"283"}
```

### Scoped Profiling

`KRONOS_TIMER("name")` creates an RAII timer that records elapsed wall time
in the global `TimerRegistry`. Timing data is included in the JSON output
and printed as a summary table at program exit.

### SCF Step Output

Each SCF iteration prints to stdout:

```
SCF step  1: E =  -27.123456 Ry  |dE| = ---        |dn| = 1.23e-02  t = 0.5s
SCF step  2: E =  -27.234567 Ry  |dE| = 1.11e-01  |dn| = 3.45e-03  t = 0.4s
```

### Error Handling

| Error condition | Response |
|----------------|----------|
| SCF non-convergence | Write partial output with `converged: false`, exit code 1 |
| Energy oscillation > 1 Ry (after step 15) | Abort SCF loop with diagnostic |
| UPF parse failure | Hard abort with `UPFParseError` (exit code 3) |
| Input validation failure | Hard abort with `InputValidationError` (exit code 2) |
| LAPACK zheev failure | Throw `runtime_error` from Davidson subspace diag |
| Negative density | Clamp to 0, renormalize to conserve total charge |
| DIIS singular matrix | Fall back to most recent density only |
