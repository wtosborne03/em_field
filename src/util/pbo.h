#ifndef PBO_H
#define PBO_H

#include "constants.h"
#include <GL/glew.h> // Add necessary includes if types like GLuint are needed directly
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct SimState;

void create_pbo(SimState *state);
void render_pbo();

#endif