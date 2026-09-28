# edge-ml-runtime-lab

What happens below the training API: a small C++ tensor/autograd engine, a
PyTorch→ONNX→INT8 deployment pipeline benchmarked from C++, and a graph optimizer
that must stay numerically equivalent.

Three connected milestones, each with a baseline, tests and measured results.

> **Status:** milestone 1 in progress. The build, the test harness and CI are in
> place; the tensor engine is being written on top of them.

## The pipeline

```text
Tensor engine  →  Model deployment  →  Graph optimization
  (TinyTensor)      (ONNX / INT8)        (rewrite passes)
```

| Milestone | Question it answers | Status |
|---|---|---|
| **1 — TinyTensor** | What does a forward/backward cost when you write it yourself, and how far is that from PyTorch on the same machine? | In progress |
| **2 — Quantized deployment** | How much memory and latency does INT8 save on this hardware, and what accuracy is lost? | Planned |
| **3 — Graph optimizer** | Do constant folding, dead-node elimination and fusion actually make a real model faster? | Planned |

Each milestone ends with a table of numbers, the conditions they were measured
under, and the cases where the result goes the other way.

## Building

Requires CMake 3.25+, a C++20 compiler, and Ninja for the `ninja` presets.
On Windows the `vs` preset locates MSVC by itself, so a plain shell is enough.

```sh
cmake --workflow --preset ninja    # configure, build, test
cmake --workflow --preset vs       # same, through Visual Studio 2022
```

Individual steps, if you want them apart:

```sh
cmake --preset ninja
cmake --build --preset ninja
ctest --preset ninja
```

CI builds and tests on Linux (GCC) and Windows (MSVC), with warnings as errors.

## Layout

```text
tinytensor/     tensor and autograd engine (milestone 1)
tests/          Catch2 suite, one CTest entry per case
benchmarks/     measurement harnesses
cmake/          shared build logic
docs/           write-ups and results
scripts/        helper scripts
```

## Scope

Deliberately fixed, so the measurements mean something:

- `float` only, CPU only, row-major contiguous tensors;
- broadcasting only where a linear layer needs it: `[n, m] + [m]`;
- no non-contiguous views, no autocast, no distributed anything;
- the goal is not to reimplement PyTorch, it is to be able to explain what it does.

## License

MIT.
