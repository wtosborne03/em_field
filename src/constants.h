#ifndef CONSTANTS_H
#define CONSTANTS_H

#define SIZE_X 1024
#define SIZE_Y 1024
#define BLOCK_SIZE 256

// Physical constants
extern const float dx = 0.01f;
extern const float mu0 = 4.0f * M_PI * 1e-7f; //permeability
extern const float eps0 = 8.854187817e-12f; //permittivity
extern const float c = 1.0f / sqrt(mu0 * eps0);
extern const float dt = 0.99f * dx / (c * sqrtf(2.0f));
extern const float CeH = dt / (mu0 * dx);
extern const float CeE = dt / (eps0 * dx);

#endif 
