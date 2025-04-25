#ifndef SIM_H
#define SIM_H

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "constants.h"
#include "kernels.cuh"

typedef struct
{
    double *Ez;
    double *Hx;
    double *Hy;
} EM_field_d;

typedef struct
{
    int mouseX, mouseY;
    bool mouseClicked;
    float amplitude;
    float boxSize;

    GLuint pbo;
    cudaGraphicsResource *cuda_pbo_resource;

    double *d_Ez, *d_Hx, *d_Hy;
    double *d_Ez_prev;
    double *d_Pec_Mask;
    EM_field_d *d_field;
} SimState;

// Function declarations
void init_gpu(SimState *state);
void display(SimState *state);
void cleanup(SimState *state);

#endif
