# CUDA 2-D FDTD electromagnetic solver (Clemson CPSC 4780)

![screenshot](https://github.com/wtosborne03/em_field/blob/main/sim_screenshot.png?raw=true)

## Project Overview

This project implements a 2-dimensional Finite-Difference Time-Domain (FDTD) method to simulate electromagnetic wave propagation. It leverages NVIDIA CUDA for parallel computation on the GPU to accelerate the simulation process. The visualization is handled using OpenGL and GLUI.

## Building and Running

### Prerequisites

*   NVIDIA CUDA Toolkit (Adjust CUDA_ARCH in the makefile based on your gpu's compute ability)
*   OpenGL development libraries (GLUT, GLEW)
*   GLUI library (included in the `libs` directory, replace with a version compiled for your system)

### Build Instructions

1.  Navigate to the project's root directory in your terminal.
2.  Run the `make` command:
    ```bash
    make
    ```
    This will compile the source files and create an executable named `em_vis`.

### Running the Simulation

1.  After a successful build, run the executable:
    ```bash
    ./em_vis
    ```
    This will launch the simulation window.

### Cleaning Up

To remove the compiled object files and the executable, run:
```bash
make clean
```

## Usage

To add a gaussian pulse, use the left mouse button. To create a circle or square of material on the grid, use the right mouse button. The side window can be used to adjust the simulation space step speeds, the amplitude and spread of the pulse, and the material type and shape type for adding materials.

## Parallelization and FDTD Method


The computationally intensive part of the FDTD simulation involves updating the electric (E) and magnetic (H) field components at each grid point for every time step. This project uses CUDA to parallelize these updates.

*   The simulation grid is mapped onto the GPU's parallel processing units.
*   CUDA kernels are launched to simultaneously calculate the field updates for a large number of grid cells. This significantly speeds up the simulation compared to a purely CPU-based implementation.
*   Memory transfers between the CPU (host) and GPU (device) are managed to update the simulation state and retrieve data for visualization, using opengl interop to visualize in real time without having to transfer the grid to host.

### FDTD Method Fundamentals

The Finite-Difference Time-Domain (FDTD) method is a technique used for modeling computational electrodynamics. It solves Maxwell's equations in their differential form, with respect to the simulation timesteps.

*   **Discretization:** Space and time are discretized into a grid (Yee grid). E and H field components are sampled at staggered locations in space and time.
*   **Update Equations:** Finite differences are used to approximate the spatial and temporal derivatives in Maxwell's curl equations. This results in update equations where the future value of a field component at a point depends only on the past values of field components at adjacent points.
    *   The H-field is updated based on the surrounding E-field values.
    *   The E-field is updated based on the surrounding H-field values.
*   **Time Stepping:** The simulation progresses by updating the E and H fields across the entire grid for discrete time steps, simulating the propagation of electromagnetic waves.
*   **Boundary Conditions:** Appropriate boundary conditions are used at the edges of the simulation domain to prevent artificial reflections.
*   **Material Properties:** Different materials (like dielectrics) are simulated by assigning specific permittivity (epsilon) and conductivity (sigma, representing loss) values to grid cells. These values modify the update equations within those cells.
