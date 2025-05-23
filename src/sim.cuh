#ifndef SIM_CUH
#define SIM_CUH

#include "constants.h"

#include <GL/glew.h>
#include <GL/glut.h>
#include <stdbool.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

typedef struct
{
    double *Ez;
    double *Hx;
    double *Hy;

    double *epsilon;
    double *mu;
    double *sigma;
} EM_field_d;

typedef struct
{
    int mouseX, mouseY;

    bool mouseClicked;
    bool shapeClicked;

    float amplitude;
    float spread;

    float boxSize;

    float dx;

    GLuint pbo;
    cudaGraphicsResource *cuda_pbo_resource;

    double *d_Ez, *d_Hx, *d_Hy;
    double *d_epsilon, *d_mu, *d_sigma;
    int *d_label;
    double *d_Ez_prev;
    EM_field_d *d_field;
    
    material *materials;
    material *d_materials;
    int selected_material;
    int num_materials;

    int shape_type;
} SimState;

#include "kernels.cuh"
#include "util/pbo.cuh"

// Function declarations
void init_gpu(SimState *state);
void display(SimState *state);
void cleanup(SimState *state);

#endif
