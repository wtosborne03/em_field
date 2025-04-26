#include "sim.cuh"

const float dx = 0.01f;
const float mu0 = 4.0f * M_PI * 1e-7f; // permeability
const float eps0 = 8.854187817e-12f;   // permittivity
const float c = 1.0f / sqrt(mu0 * eps0);
const float dt = 0.99f * dx / (c * sqrtf(2.0f));
const float CeH = dt / (mu0 * dx);
const float CeE = dt / (eps0 * dx);

void init_gpu(SimState *state)
{
    size_t total_size = SIZE_X * SIZE_Y * sizeof(double);

    cudaMalloc(&(state->d_Ez), total_size);
    cudaMalloc(&(state->d_Hx), total_size);
    cudaMalloc(&(state->d_Hy), total_size);

    cudaMalloc(&(state->d_Ez_prev), total_size);  // same size as Ez
    cudaMalloc(&(state->d_Pec_Mask), total_size); // same size as Ez

    cudaMemset(state->d_Ez, 0, total_size);
    cudaMemset(state->d_Hx, 0, total_size);
    cudaMemset(state->d_Hy, 0, total_size);

    cudaMemset(state->d_Ez_prev, 0, total_size);
    cudaMemset(state->d_Pec_Mask, 0, total_size);

    cudaMalloc(&(state->d_field), sizeof(EM_field_d));
}

void display(SimState *state)
{
    size_t total_size = SIZE_X * SIZE_Y * sizeof(double);

    cudaMemcpy(state->d_Ez_prev, state->d_Ez, total_size, cudaMemcpyDeviceToDevice);

    dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);

    updateH<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, CeH);
    updateE<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, CeE);
    apply_damping<<<(SIZE_X * SIZE_Y + 255) / 256, 256>>>(state->d_field, SIZE_X * SIZE_Y, 0.995f);

    mur_boundary<<<(SIZE_X * SIZE_Y + 255) / 256, 256>>>(state->d_Ez, SIZE_X, SIZE_Y, (c * dt / dx), state->d_Ez_prev);
    apply_pec_mask<<<(SIZE_X * SIZE_Y + 255) / 256, 256>>>(state->d_field, state->d_Pec_Mask, SIZE_X * SIZE_Y);

    if (state->mouseClicked) {
            dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);
            gaussian_pulse<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y,state->mouseX, SIZE_Y - state->mouseY, state->amplitude, 10.0f);
    }

    float *d_pbo;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &(state->cuda_pbo_resource), 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &(state->num_bytes), state->cuda_pbo_resource);
    write_to_pbo<<<(SIZE_X * SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(state->d_field, state->d_Pec_Mask, d_pbo, SIZE_X * SIZE_Y);
    cudaGraphicsUnmapResources(1, &(state->cuda_pbo_resource), 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, state->pbo);
    render_pbo();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();
}

// deallocate before exiting
void cleanup(SimState *state)
{
    cudaFree(state->d_Ez);
    cudaFree(state->d_Hx);
    cudaFree(state->d_Hy);
    cudaFree(state->d_field);
    cudaFree(state->d_Ez_prev);
    cudaFree(state->d_Pec_Mask);
    cudaGraphicsUnregisterResource(state->cuda_pbo_resource);
    glDeleteBuffers(1, &(state->pbo));
}