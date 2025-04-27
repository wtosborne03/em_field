#include "pbo.cuh"

void create_pbo(SimState *state)
{
    glGenBuffers(1, &(state->pbo));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, state->pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, SIZE_X * SIZE_Y * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&(state->cuda_pbo_resource), state->pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void render_pbo()
{
    glRasterPos2i(-1, -1);
    glDrawPixels(SIZE_X, SIZE_Y, GL_RGB, GL_FLOAT, 0);
}