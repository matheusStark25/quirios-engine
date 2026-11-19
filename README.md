# QURIOS Engine

**Infrastructure between classical and quantum software.** An ordinary program talks to QURIOS through
interfaces it already knows (`fit`/`predict` in Python, or JSON over HTTP). QURIOS builds the quantum circuits,
picks where to run them (exact simulation, tensor networks, or a real quantum processor), decides **with
statistics** when quantum is worth it and when classical is better, and returns a signed proof of every
execution.

[![tests](https://github.com/matheusStark25/quirios-engine/actions/workflows/testes.yml/badge.svg)](https://github.com/matheusStark25/quirios-engine/actions/workflows/testes.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey)

> 🇧🇷 **Documentação completa em português:** [README.pt-BR.md](README.pt-BR.md) ·
> [ARQUITETURA.md](ARQUITETURA.md) · [SEGURANCA.md](SEGURANCA.md) · [CHANGELOG.md](CHANGELOG.md)

---

## Measured on real quantum hardware

On **two IBM quantum processors** (Heron, 156 superconducting qubits each), QURIOS learned a property of
quantum states **with quantum memory** (two copies stored and measured together) and was compared against the
best classical strategy (each copy measured as soon as it arrives), **with the same number of copies on both
sides**.

| Processor | Run | Test states | With quantum memory | Without (classical) | Paired McNemar test |
|---|---|---|---|---|---|
| ibm_marrakesh | 1 | 20 | 80% | 70% | 4 vs 2, p = 0.69 |
| ibm_marrakesh | 2 | 40 | 97.5% | 77.5% | 9 vs 1, p = 0.021 |
| ibm_kingston | 3 (new states) | 40 | 95% | 60% | 15 vs 1, p = 0.0005 |
| **Both chips** | | **100** | **93%** (95% CI 86–97%) | **69%** (95% CI 59–77%) | **28 vs 4, p = 0.00002** |

*4 qubits, 64 copies per state for each side, November 19, 2026. 12 jobs, 39 seconds of QPU time in total.*

- **Why McNemar:** both methods answer about the *same* states, so only the states where they disagree count.
  Out of 32 disagreements, quantum memory was right 28 times.
- **Replicated:** the first run alone was not significant. The advantage then held on a second chip, with
  freshly drawn states and the success criterion fixed *before* the run.
- **The simulation predicted the chip:** QURIOS' noisy rehearsal of ibm_kingston predicted 97.5% vs 57.5%;
  the real chip gave 95% vs 60%, with the same calibrated signal levels (0.95 and 0.85).
- **The science:** this is the family of tasks in which Huang et al. (*Science* 376, 1182, 2022) proved an
  exponential separation between learning with and without quantum memory. QURIOS turns it into a
  one-command, verifiable infrastructure feature.

### Verify it yourself

Every run produces a proof: the IBM job ids, backend, timestamps, QPU time, SHA-256 of the raw counts and the
circuits in OpenQASM 3, sealed and signed with **ML-DSA-65** (NIST FIPS 204, post-quantum). The proofs and the
public key are in [`resultados/qpu_real/`](resultados/qpu_real/):

```bash
pip install -e ".[qiskit,pqc]"
python -m quirios_engine.qpu verificar resultados/qpu_real/comprovante_qpu_memoria_ibm_kingston_20261119_194200_calibrado.json --chave-publica resultados/qpu_real/chave_publica_mldsa65.pub.pem
```

Change a single number in a proof and the check fails. The job ids (for example `dbvnmq5s2omc73886k00` and
`dbvnnh5s2omc73886lgg` on ibm_kingston) can be confirmed in IBM's dashboard by the account owner.

### What this is not

A laboratory task on 4 qubits (deciding whether a quantum state is time-reversal symmetric), not a product
yet. The advantage exists when the data is **born quantum** (a sensor, a quantum simulator, another quantum
processor) and reaches the processor without being measured first. It does **not** speed up classical data.

---

## Everything else, with where it was measured

| Result | Measured on |
|---|---|
| Quantum memory, 6 qubits: **98% vs 49%** for the best classical method, same number of copies | Exact simulation |
| AI layer written as a tensor network (the math of quantum states): MNIST model **35× smaller**, 97.46% vs 97.34% for the original | CPU (PyTorch) |
| Same accuracy with **9–22× fewer parameters** than classical low-rank compression; wide layers **6–30× faster** at batch size 1 | CPU (PyTorch) |
| Loading a classical 128×128 photo into a quantum processor with **68× fewer two-qubit gates** (240 vs 16,369 CNOTs, fidelity 0.952) | Compiled circuits |
| Full proof run on the real chip: data prepared and classified on the QPU **tied** classical (80% vs 80%); architecture evolution sampled on the QPU reached 92.5% vs 88.7% for the default circuit (the winner uses no two-qubit gates) | ibm_marrakesh |
| Native tensor-network language model: beats classical low-rank compression (3.94 vs 4.24 bits/char) but **loses** to a narrower dense model (3.71) at this scale | CPU |
| Ising phases with the same measurement budget for both sides: **classical wins** (−6 to −11 points) | Simulation |

QURIOS reports where it loses as clearly as where it wins. An earlier "+18 points" result was **retracted**
when it turned out the comparison was unfair ([CHANGELOG](CHANGELOG.md)).

---

## What QURIOS does

```
 AI agents (MCP) ──┐
 classical software ┴─► HTTP API / Python
                         │
                   Orchestrator ── classical or quantum? (cross-validation + statistical test)
                         │
        Auto-evolution ── architecture search, champions promoted only with evidence
                         │
   Training / kernels ── parameter-shift | SPSA | projected quantum kernels
                         │
     Backends ── exact (≤24 qubits) | MPS (50–100+) | Aer | IBM Quantum | IQM Resonance
                         │
     Proofs ── signed with post-quantum cryptography (ML-DSA-65)
```

- **One contract, many backends:** IBM Quantum today and IQM Resonance (Finland) prepared; switching vendor is
  one parameter.
- **Evidence before promotion:** a quantum model replaces a classical one only when a statistical test says so.
- **Security built in:** the API refuses to start without a key; scoped, hashed keys; rate limiting and
  brute-force lockout; a hash-chained audit trail with a post-quantum signed head; models stored encrypted with
  ML-KEM-768 + AES-256-GCM; an MCP guard for AI agents. Details in [SEGURANCA.md](SEGURANCA.md).
- **329 automated tests**, including attacks on every security layer, end-to-end runs against a real HTTP
  server, and stress tests.

## Quick start

```bash
pip install -e ".[qiskit,api,pqc]"

# full proof in a local rehearsal (simulator with a real chip's noise model, no account needed)
python -m quirios_engine.qpu prova --modo ruidoso

# on a real IBM quantum processor (free Open Plan: 10 QPU minutes every 28 days)
python -m quirios_engine.conta_ibm          # opens a small window to paste your API key; it is never shown
python -m quirios_engine.qpu prova          # picks the least busy QPU, asks once, signs every proof
```

The HTTP API, the MCP server, the tensor-network AI tools and the Kaggle notebook are documented in
[README.pt-BR.md](README.pt-BR.md).

## License

**Copyright © 2026 Matheus Willian Biagioni Assis. All rights reserved.** The code is public so that every
result above can be checked. Using, copying, modifying or distributing it requires written permission: see
[LICENSE](LICENSE).

Built by [Matheus Assis](https://github.com/matheusStark25).
