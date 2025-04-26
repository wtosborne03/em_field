#ifndef PBO_H
#define PBO_H

#include "sim.cuh"
#include "constants.h"

void create_pbo(SimState *state);
void render_pbo();

#endif