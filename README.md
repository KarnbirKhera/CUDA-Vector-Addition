# CUDA Vector Sum Learning Report

Teaching myself CUDA through vector addition by writing my first kernel to understanding register pressure, SM occupancy, memory coalescing, and using Nsight Compute to figure out why my "optimizations" made things slower.

<h2>Overview</h2>

This project implements five progressively "optimized" CUDA kernels for vector addition, benchmarks them, and analyzes why the naive implementation won in this case.



