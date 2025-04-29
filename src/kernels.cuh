#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "sim.cuh"

__global__ void updateE(EM_field_d *field, int width, int height, double dt, double dx);
__global__ void updateH(EM_field_d *field, int width, int height, double dt, double dx);
__global__ void write_to_pbo(EM_field_d *field,  int * d_label,material * materials, float *pbo, int n);
__global__ void apply_damping(EM_field_d *field, int n, double factor);
__global__ void mur_boundary(double* Ez, int width, int height, float cdt_dx, double* Ez_prev);
__global__ void gaussian_pulse(EM_field_d * field, int w, int h, int cx, int cy, float A, float sigma);
__global__ void add_box(EM_field_d * field, int * d_label, int mat, material * materials, int cx, int cy, int size, int width, int height);

#endif