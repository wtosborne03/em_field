#include "sim.cuh"

const float mu0 = 4.0f * M_PI * 1e-7f; // permeability
const float eps0 = 8.854187817e-12f;   // permittivity
const float c = 1.0f / sqrt(mu0 * eps0);

void init_gpu(SimState *state)
{
    size_t total_size = SIZE_X * SIZE_Y * sizeof(double);
    size_t materials_size = state->num_materials * sizeof(material);

    cudaMalloc(&(state->d_Ez), total_size);
    cudaMalloc(&(state->d_Hx), total_size);
    cudaMalloc(&(state->d_Hy), total_size);
    // material grid alloc
    cudaMalloc(&(state->d_epsilon), total_size);
    cudaMalloc(&(state->d_mu), total_size);
    cudaMalloc(&(state->d_sigma), total_size);

    cudaMalloc(&(state->d_Ez_prev), total_size); // same size as Ez
    cudaMalloc(&(state->d_label), SIZE_X * SIZE_Y * sizeof(int));
    cudaMalloc(&(state->d_materials), materials_size);

    // memset
    cudaMemset(state->d_Ez, 0, total_size);
    cudaMemset(state->d_Hx, 0, total_size);
    cudaMemset(state->d_Hy, 0, total_size);

    cudaMemset(state->d_epsilon, 0, total_size);
    cudaMemset(state->d_mu, 0, total_size);
    cudaMemset(state->d_sigma, 0, total_size);

    double *h_epsilon, *h_mu, *h_sigma;
    h_epsilon = (double *)malloc(total_size);
    h_mu = (double *)malloc(total_size);
    h_sigma = (double *)malloc(total_size);

    // need to set initial material arrays on host before copying to device
    // needed for simulating air at start
    for (int i = 0; i < SIZE_X * SIZE_Y; i++)
    {
        h_epsilon[i] = eps0;
        h_mu[i] = mu0;
        h_sigma[i] = 0.0;
    }

    cudaMemcpy(state->d_epsilon, h_epsilon, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_mu, h_mu, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(state->d_sigma, h_sigma, total_size, cudaMemcpyHostToDevice);

    cudaMemcpy(state->d_materials, state->materials, materials_size, cudaMemcpyHostToDevice);

    cudaMemset(state->d_Ez_prev, 0, total_size);
    cudaMalloc(&(state->d_field), sizeof(EM_field_d));

    cudaMemset(state->d_label, 0, total_size);

    EM_field_d h_field; // Host-side temporary struct
    h_field.Ez = state->d_Ez;
    h_field.Hx = state->d_Hx;
    h_field.Hy = state->d_Hy;

    h_field.epsilon = state->d_epsilon;
    h_field.mu = state->d_mu;
    h_field.sigma = state->d_sigma;

    cudaMemcpy(state->d_field, &h_field, sizeof(EM_field_d), cudaMemcpyHostToDevice);
    free(h_epsilon);
    free(h_mu);
    free(h_sigma);
}

void display(SimState *state)
{
    float dx = state->dx;
    float dt = 0.99f * dx / (c * sqrtf(2.0f));

    size_t total_size = SIZE_X * SIZE_Y * sizeof(double);

    cudaMemcpy(state->d_Ez_prev, state->d_Ez, total_size, cudaMemcpyDeviceToDevice);

    dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);

    updateH<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, dt, dx);
    updateE<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, dt, dx);
    apply_damping<<<(SIZE_X * SIZE_Y + 255) / 256, 256>>>(state->d_field, SIZE_X * SIZE_Y, 0.995f);

    mur_boundary<<<(SIZE_Y + 255) / 256, 256>>>(state->d_Ez, SIZE_X, SIZE_Y, (c * dt / dx), state->d_Ez_prev);

    if (state->mouseClicked)
    {
        dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);
        gaussian_pulse<<<grid, block>>>(state->d_field, SIZE_X, SIZE_Y, state->mouseX, SIZE_Y - state->mouseY, state->amplitude, 10.0f);
    }
    float *d_pbo;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &(state->cuda_pbo_resource), 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &num_bytes, state->cuda_pbo_resource);
    write_to_pbo<<<(SIZE_X * SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(state->d_field, state->d_label, state->materials, d_pbo, SIZE_X * SIZE_Y);
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

    cudaFree(state->d_label);
    cudaFree(state->d_materials);

    cudaFree(state->d_field);
    cudaFree(state->d_Ez_prev);
    cudaGraphicsUnregisterResource(state->cuda_pbo_resource);
    glDeleteBuffers(1, &(state->pbo));
}