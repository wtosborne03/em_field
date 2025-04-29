#include "kernels.cuh"

__global__ void updateE(EM_field_d *field, int width, int height, double dt, double dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // bounds check
    if (x < 1 || y < 1) return;

    int idx = y * width + x;

    double eps = field->epsilon[idx];
    eps = (fabs(eps) > 1e-15) ? eps : 1e-15 * (eps < 0 ? -1.0 : 1.0);

    double eq1 = ((1.0 - (field->sigma[idx] * dt) / (2.0 * eps)) /
              (1.0 + (field->sigma[idx] * dt) / (2.0 * eps))) * field->Ez[idx];

    double eq2 = (dt / (eps * (1.0 + (field->sigma[idx] * dt) / (2.0 * eps))));

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

    double mu = field->mu[idx];
    mu = (fabs(mu) > 1e-15 ? mu : 1e-15 * (mu < 0 ? -1.0 : 1.0));

    double CeH = dt / (mu * dx);

    field->Hx[idx] -= CeH * (field->Ez[idx + width] - field->Ez[idx]);
    field->Hy[idx] += CeH * (field->Ez[idx + 1] - field->Ez[idx]);

}

__global__ void clamp_fields(EM_field_d *field, int n, double maxval) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    field->Ez[idx] = fmax(fmin(field->Ez[idx], maxval), -maxval);
    field->Hx[idx] = fmax(fmin(field->Hx[idx], maxval), -maxval);
    field->Hy[idx] = fmax(fmin(field->Hy[idx], maxval), -maxval);
}


// copy Ez to the PBO
__global__ void write_to_pbo(EM_field_d *field, int *d_label, material *materials, float *pbo, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = fmaxf(fminf(field->Ez[idx], 1.0f), -1.0f);
    
    // Base color from Ez (blue/red)
    float base_r = (v > 0.0f) ? v : 0.0f;
    float base_g = 0.0f;
    float base_b = (v < 0.0f) ? -v : 0.0f;

    if (d_label[idx] != 0) {
        // Tint the base color with the material color
        float tint = 0.3f;
        float mat_r = materials[d_label[idx]].v_r;
        float mat_g = materials[d_label[idx]].v_g;
        float mat_b = materials[d_label[idx]].v_b;

        pbo[3 * idx + 0] = (1.0f - tint) * base_r + tint * mat_r;
        pbo[3 * idx + 1] = (1.0f - tint) * base_g + tint * mat_g;
        pbo[3 * idx + 2] = (1.0f - tint) * base_b + tint * mat_b;
    } else {
        // Normal wave color
        pbo[3 * idx + 0] = base_r;
        pbo[3 * idx + 1] = base_g;
        pbo[3 * idx + 2] = base_b;
    }
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


// shape kernels
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

__global__ void add_circle(EM_field_d * field, int * d_label, int mat, material * materials, int cx, int cy, int size, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half = size / 2;

    if (sqrtf(pow(x - cx, 2) + pow(y - cy,2)) < size)
    {
        int idx = y * width + x;
        field->epsilon[idx] =  materials[mat].permittivity;
        field->mu[idx] = materials[mat].permeability;
        field->sigma[idx] = materials[mat].conductivity;
        d_label[idx] = mat;
    }
    
}

