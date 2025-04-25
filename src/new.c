#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glui.h>
#include <stdio.h>
#include <math.h>

#include "pbo.h"

#define SIZE_X 1024
#define SIZE_Y 1024
#define BLOCK_SIZE 256

typedef struct {
    int mouseX, mouseY;
    bool mouseClicked;
    float amplitude;
    float boxSize;

    GLuint pbo;
    cudaGraphicsResource *cuda_pbo_resource;

    double *d_Ez, *d_Hx, *d_Hy;
    double *d_Ez_prev;
    double *d_Pec_Mask;
    EM_field_d *d_field;
} SimState;

static SimState* global_state = NULL;

const float dx = 0.01f;
const float mu0 = 4.0f * M_PI * 1e-7f;
const float eps0 = 8.854187817e-12f;
const float c = 1.0f / sqrt(mu0 * eps0);
const float dt = 0.99f * dx / (c * sqrtf(2.0f));
const float CeH = dt / (mu0 * dx);
const float CeE = dt / (eps0 * dx);

// Declare your CUDA kernels here or include them from a separate file...
// For brevity, not duplicating kernel code in this snippet.

void display_wrapper() {
    display(global_state);
}

void keyboard_wrapper(unsigned char key, int x, int y) {
    if (key == 27) {
        cleanup(global_state);
        exit(0);
    }
}

void mouse_func_wrapper(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);
        gaussian_pulse<<<grid, block>>>(global_state->d_field, SIZE_X, SIZE_Y, x, SIZE_Y - y, global_state->amplitude, 10.0f);
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        dim3 block(16, 16), grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);
        add_box<<<grid, block>>>(global_state->d_Pec_Mask, x, SIZE_Y - y, global_state->boxSize, SIZE_X, SIZE_Y);
    }
}

void control_cb(int id) {}

int main(int argc, char** argv) {
    static SimState state = {
        .mouseX = -1,
        .mouseY = -1,
        .mouseClicked = false,
        .amplitude = 10.0f,
        .boxSize = 50.0f
    };
    global_state = &state;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(SIZE_X, SIZE_Y);
    int main_window = glutCreateWindow("CUDA EM Sim");

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
        return 1;
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glViewport(0, 0, SIZE_X, SIZE_Y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    size_t total_size = SIZE_X * SIZE_Y * sizeof(double);
    cudaMalloc(&state.d_Ez, total_size);
    cudaMalloc(&state.d_Hx, total_size);
    cudaMalloc(&state.d_Hy, total_size);
    cudaMalloc(&state.d_Ez_prev, total_size);
    cudaMalloc(&state.d_Pec_Mask, total_size);

    EM_field_d h_field_dev = {
        .Ez = state.d_Ez,
        .Hx = state.d_Hx,
        .Hy = state.d_Hy
    };

    cudaMemset(state.d_Ez, 0, total_size);
    cudaMemset(state.d_Hx, 0, total_size);
    cudaMemset(state.d_Hy, 0, total_size);
    cudaMemset(state.d_Ez_prev, 0, total_size);

    cudaMalloc(&state.d_field, sizeof(EM_field_d));
    cudaMemcpy(state.d_field, &h_field_dev, sizeof(EM_field_d), cudaMemcpyHostToDevice);

    create_pbo(&state.pbo, SIZE_X, SIZE_Y, state.cuda_pbo_resource);

    GLUI *glui = GLUI_Master.create_glui("Controls");
    glui->add_spinner("Amplitude", GLUI_SPINNER_FLOAT, &state.amplitude, 1, control_cb)->set_float_limits(2.0, 50.0);
    glui->add_spinner("Box_Size", GLUI_SPINNER_FLOAT, &state.boxSize, 1, control_cb)->set_float_limits(20.0, 200.0);
    glui->set_main_gfx_window(main_window);

    glutDisplayFunc(display_wrapper);
    glutKeyboardFunc(keyboard_wrapper);
    glutMouseFunc(mouse_func_wrapper);

    glutMainLoop();
    return 0;
}
