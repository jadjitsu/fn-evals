# fn-evals (v0.1)

**Problem →** Early-phase teams often lack tiny, fast, *reproducible* sanity-benchmarks to validate pipelines on CPU-only hardware.

**Method →** Provide a micro-eval harness that runs in under 30 seconds on CPU, emits a machine-parsable CSV row (runtime, memory, correctness checks), and is CI-verified.

**Evidence →** `bench.csv` records measured runtime and correctness flags under fixed seeds/hyperparams on CPU.

---

## Reproduce (CPU, Python 3.11)

```bash
bash repro.sh
