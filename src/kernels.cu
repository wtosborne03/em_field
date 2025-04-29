#include "kernels.cuh"

__global__ void updateE(EM_field_d *field, int width, int height, double dt, double dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // bounds check
    if (x < 1 || y < 1) return;

    int idx = y * width + x;

    double eq1 = ((1.0 - (field->sigma[idx] * dt) / (2.0 * field->epsilon[idx])) /
              (1.0 + (field->sigma[idx] * dt) / (2.0 * field->epsilon[idx]))) * field->Ez[idx];

    double eq2 = (dt / (field->epsilon[idx] * (1.0 + (field->sigma[idx] * dt) / (2.0 * field->epsilon[idx]))));

    double curl = (
        (field->Hy[idx] - field->Hy[idx - 1]) -
        (field->Hx[idx] - field->Hx[idx - width])
    );

    field->Ez[idx] = (eq1 + (eq2 * curl));
}

__global__ void updateH(EM_field_d *field, int width, int height, double dt, double dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width -1 || y >= height - 1) return; // bounds check

    int idx = y * width + x;

    double CeH = dt / (field->mu[idx] * dx);

    field->Hx[idx] -= CeH * (field->Ez[idx + width] - field->Ez[idx]);
    field->Hy[idx] += CeH * (field->Ez[idx + 1] - field->Ez[idx]);

}

// copy Ez to the PBO
__global__ void write_to_pbo(EM_field_d *field, int * d_label, material * materials, float *pbo, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = fmaxf(fminf(field->Ez[idx], 1.0f), -1.0f);

    if (d_label[idx] != 0) {
        pbo[3 * idx + 0] = materials[d_label[idx]].v_r;
        pbo[3 * idx + 1] = materials[d_label[idx]].v_g;
        pbo[3 * idx + 2] = materials[d_label[idx]].v_b;
        return;
    }

    pbo[3 * idx + 0] = (v > 0.0f) ? v : 0.0f;
    pbo[3 * idx + 1] = (float)d_label[idx];
    pbo[3 * idx + 2] = (v < 0.0f) ? -v: 0.0f;

}
__global__ void apply_damping(EM_field_d *field,
                              int n, double factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    field->Ez[idx] *= factor;
    field->Hx[idx] *= factor;
    field->Hy[idx] *= factor;
}

__global__ void mur_boundary(double* Ez, int width, int height,
                             float cdt_dx, double* Ez_prev) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float coef = (cdt_dx - 1.0f) / (cdt_dx + 1.0f);

    if (j >= height) return;

    // Left boundary (i = 0)
    int idx0 = j * width;
    int idx1 = j * width + 1;
    Ez[idx0] = Ez_prev[idx1] + coef * (Ez[idx1] - Ez_prev[idx0]);

    // Right boundary (i = width - 1)
    int idxN = j * width + (width - 1);
    int idxNm1 = j * width + (width - 2);
    Ez[idxN] = Ez_prev[idxNm1] + coef * (Ez[idxNm1] - Ez_prev[idxN]);
}



__global__ void gaussian_pulse(EM_field_d * field, int w, int h, int cx, int cy, float A, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float dx = x - cx;
    float dy = y - cy;
    float r2 = dx * dx + dy * dy;
    field->Ez[y * w + x] += A * expf(-r2 / (2.0f * sigma * sigma));
}

__global__ void add_box(EM_field_d * field, int * d_label, int mat, material * materials, int cx, int cy, int size, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half = size / 2;

    if (x >= cx - half && x < cx + half &&
        y >= cy - half && y < cy + half)
    {
        int idx = y * width + x;
        field->epsilon[idx] =  materials[mat].permittivity;
        field->mu[idx] = materials[mat].permeability;
        field->sigma[idx] = materials[mat].conductivity;
        d_label[idx] = mat;
    }
    
}