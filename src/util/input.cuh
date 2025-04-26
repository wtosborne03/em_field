#ifndef INPUT_CUH
#define INPUT_CUH

#include "../constants.h"
#include "../sim.cuh"
#include <cuda_runtime.h>

void keyboard(unsigned char key, int x, int y);
void mouse_func(int button, int state, int x, int y);
void motion_func(int x, int y);
void control_cb(int id);

// Must set this before calling wrappers
void set_sim_state(SimState *state);

#endif
