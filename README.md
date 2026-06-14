# Praktikum 1 – Matrix Multiplication on CPU

> **AI Accelerators (AIA) – Lab Assignment**

Matrix multiplication is the dominant kernel in machine learning. All deep learning frameworks rely on it through heavily optimised libraries. In this lab you will start from a **naive C implementation** and apply a sequence of classical CPU optimisations to understand how hardware works and why these optimisations matter.

---

## Learning Objectives

By the end of this lab you will be able to:

- Quantify the *theoretical peak performance* of a CPU (cores × clock × SIMD × MACs).
- Explain how *loop ordering* affects cache behaviour and choose the best access pattern.
- Apply *compiler vectorisation hints* to exploit SIMD units automatically.
- Implement *loop tiling (cache blocking)* to improve temporal locality.
- Parallelise matrix multiplication and analyse scaling efficiency.
- Compare your custom C implementation with PyTorch's optimised BLAS backend.

---

## Repository Structure

```
.
├── matmul.c              ← Your main work file (Tasks 2–5 are stubs here)
├── benchmark_pytorch.py  ← PyTorch baseline benchmark
├── Makefile              ← Build system
├── Report.md             ← Fill this in for evaluation
├── figures/              ← Put your plots or figures here
```

---



## Tasks

Open **`matmul.c`** and work through the functions in order. Each function has a clearly marked `/* TODO */` comment. Do **not** modify `matmul_naive` — it is the correctness reference.

### Task 1 – System Characterisation
Fill in your hardware details in **`Report.md`** (Task 1 section). Use `lscpu`, `lstopo`, `/proc/cpuinfo`. Calculate the theoretical peak GFLOP/s.

### Task 2 – Loop Reordering (`matmul_reordered`)
Implement the best loop ordering you find. Try i-k-j, j-k-i, k-i-j, and others. Record results in the report.

### Task 3 – Vectorization (`matmul_vectorized`)
Add compiler flags `CFLAGS` in the Makefile (e.g. -Wall -O3 -ffast-math -mcpu=apple-m1). Record which flags make a measurable difference.

### Task 4 – Loop Tiling (`matmul_tiled`)
Implement cache-blocked matrix multiplication with the `TILE` macro. Tune the tile size to match your L1/L2 cache.

### Task 5 – Multithreading (`matmul_parallel`)
Parallelise with OpenMP or Pthreads. Combine with tiling. Measure scaling from 1 to max physical cores.

### Task 6 – Performance Analysis
In `Report.md`: write the achieved performance in GFLOPS

### Task 7 – Key Takeaways
Write a short reflection (3–5 sentences) in `Report.md`.

---

## Getting Started

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt install gcc make python3-pip
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Check your CPU topology
lscpu
lstopo        # (install hwloc if missing: sudo apt install hwloc)
```

### Build & Run

```bash
make                      # compile with -O3 -march=native -ffast-math
./matmul 512              # run all variants for a 512×512 matrix
make benchmark            # sweep over sizes 64 … 4096
make pytorch              # run PyTorch baseline

```

---

## Evaluation Criteria

* Code correctness (all verify checks pass) 
* Report quality and analysis depth 
* Team presentation (5–10 min)

Autograding will check that your code **compiles** and that the **correctness checks** pass for sizes 64, 128, 256, and 512, 1024.

---

## Tips

- Always verify correctness before optimising further.
- Commit and push regularly — only what is on `main` at the deadline will be graded.

---


