#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <math.h>

#define SIZE_X 1024
#define SIZE_Y 1024
#define BLOCK_SIZE 256

// defines materials
typedef struct
{
    double permittivity;
    double permeability;
    double conductivity;

    float v_r;
    float v_g;
    float v_b;

    char name[50];
} material;

#endif
