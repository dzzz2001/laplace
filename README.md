# 2D Laplace Equation Solver using MPI and CUDA Hybrid Programming

## Overview

This project implements an iterative solver for the two-dimensional Laplace equation using a hybrid MPI and CUDA programming approach. It supports multi-GPU computations, allowing for efficient parallel processing across multiple GPUs.

## Features

- Iterative solution of the 2D Laplace equation.
- Hybrid MPI and CUDA programming for both intra-node GPU acceleration and inter-node communication.
- Support for multi-GPU configurations.
- Choice between Jacobi and Gauss-Seidel iterative methods.

## Compilation

To compile the program, follow these steps:

1. Ensure you have CMake installed.
2. Run the following commands in your terminal:

```bash
cmake -B build
cmake --build build
```

## Running the Program
The program requires six input parameters to run. Execute the program as follows:

```bash
mpirun -np <num_processes> ./your_executable <param1> <param2> <param3> <param4> <param5> <param6>
```
### parameters

1. Grid Points in X Direction (\<param1\>): Number of grid points along the x-axis.
2. Grid Points in Y Direction (\<param2\>): Number of grid points along the y-axis.
3. Partitioning in X Direction (\<param3\>): Number of partitions along the x-axis.
4. Partitioning in Y Direction (\<param4\>): Number of partitions along the y-axis.
5. Maximum Iterations (\<param5\>): Maximum number of iterations allowed for the solver.
6. Iterative Method (\<param6\>):
  -Input 0 to select the Jacobi iterative method.
  -Input 1 to select the Gauss-Seidel iterative method.
Note: Ensure that the product of \<param3\> (x-direction partitioning) and \<param4\> (y-direction partitioning) is less than the actual number of processes (\<num_processes\>).

## Example usage

Assuming you want to run the program with 4 processes, 100 grid points in each direction, 2 partitions in the x-direction, 2 partitions in the y-direction, a maximum of 1000 iterations, and using the Jacobi method:

```bash
mpirun -np 4 ./your_executable 100 100 2 2 1000 0
```
