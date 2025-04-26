#include "kernels.cuh"

__global__ void updateE(EM_field_d *field, int width, int height, double CeE) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // bounds check
    if (x < 1 || y < 1) return;

    int idx = y * width + x;

    field->Ez[idx] += CeE * (
        (field->Hy[idx] - field->Hy[idx - 1]) -
        (field->Hx[idx] - field->Hx[idx - width])
    );
}

__global__ void updateH(EM_field_d *field, int width, int height, double CeH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width -1 || y >= height - 1) return; // bounds check

    int idx = y * width + x;


    field->Hx[idx] -= CeH * (field->Ez[idx + width] - field->Ez[idx]);
    field->Hy[idx] += CeH * (field->Ez[idx + 1] - field->Ez[idx]);

}

// copy Ez to the PBO
__global__ void write_to_pbo(EM_field_d *field, double * pec_mask, float *pbo, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = fmaxf(fminf(field->Ez[idx], 1.0f), -1.0f);

    pbo[3 * idx + 0] = (v > 0.0f) ? v : 0.0f;
    pbo[3 * idx + 1] = (pec_mask[idx] > 0.5f) ? 1.0f : 0.0f;
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

__global__ void add_box(double * pec_mask, int cx, int cy, int size, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half = size / 2;

    if (x >= cx - half && x < cx + half &&
        y >= cy - half && y < cy + half)
    {
        int idx = y * width + x;
        pec_mask[idx] = 1.0f;
    }
    

}

__global__ void apply_pec_mask(EM_field_d * field, double * pec_mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if(pec_mask[idx] > 0.5f) {
        field->Ez[idx] = 0.0f;
    }
}