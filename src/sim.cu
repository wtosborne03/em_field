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
    // material grid alloc
    cudaMalloc(&(state->d_epsilon), total_size);
    cudaMalloc(&(state->d_mu), total_size);
    cudaMalloc(&(state->d_sigma), total_size);


    cudaMalloc(&(state->d_Ez_prev), total_size);  // same size as Ez

    //memset
    cudaMemset(state->d_Ez, 0, total_size);
    cudaMemset(state->d_Hx, 0, total_size);
    cudaMemset(state->d_Hy, 0, total_size);



    cudaMemset(state->d_epsilon, 0, total_size);
    cudaMemset(state->d_mu, 0, total_size);
    cudaMemset(state->d_sigma, 0, total_size);

    cudaMemset(state->d_Ez_prev, 0, total_size);
    cudaMalloc(&(state->d_field), sizeof(EM_field_d));

    EM_field_d h_field; // Host-side temporary struct
    h_field.Ez = state->d_Ez;
    h_field.Hx = state->d_Hx;
    h_field.Hy = state->d_Hy;
    h_field.epsilon = state->d_epsilon;
    h_field.mu = state->d_mu;
    h_field.sigma = state->d_sigma;

    // need to set initial material arrays on host before copying to device
    // needed for simulating air at start
    for (int i = 0; i < SIZE_X; i++) {
        for (int j = 0; j < SIZE_Y; j++) {
            h_field.epsilon[i * SIZE_X + j] = eps0;
            h_field.mu[i * SIZE_X + j] = mu0;
            h_field.sigma[i * SIZE_X + j] = 0.0;
        }
    }

    cudaMemcpy(state->d_field, &h_field, sizeof(EM_field_d), cudaMemcpyHostToDevice);
}

void display(SimState *state)
{
    size_t total_size = SIZE_X * SIZE_Y * sizeof(double);

    cudaMemcpy(state->d_Ez_prev, state->d_Ez, total_size, cudaMemcpyDeviceToDevice);

    dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);

    updateH<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, dt, dx);
    updateE<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, dt, dx);
    apply_damping<<<(SIZE_X * SIZE_Y + 255) / 256, 256>>>(state->d_field, SIZE_X * SIZE_Y, 0.995f);

    mur_boundary<<<(SIZE_X * SIZE_Y + 255) / 256, 256>>>(state->d_Ez, SIZE_X, SIZE_Y, (c * dt / dx), state->d_Ez_prev);

    if (state->mouseClicked)
    {
        dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);
        gaussian_pulse<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, state->mouseX, SIZE_Y - state->mouseY, state->amplitude, 10.0f);
    }
    printf("MouseX: %i, MouseY: %i\n", state->mouseX, state->mouseY);

    float *d_pbo;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &(state->cuda_pbo_resource), 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &num_bytes, state->cuda_pbo_resource);
    write_to_pbo<<<(SIZE_X * SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(state->d_field,  d_pbo, SIZE_X * SIZE_Y);
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
    cudaFree(state->d_epsilon);
    cudaFree(state->d_mu);
    cudaFree(state->d_sigma);

    cudaFree(state->d_field);
    cudaFree(state->d_Ez_prev);
    cudaGraphicsUnregisterResource(state->cuda_pbo_resource);
    glDeleteBuffers(1, &(state->pbo));
}