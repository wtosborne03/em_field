#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <GL/glui.h>

#include "util/pbo.h"
#include "util/input.h"
#include "constants.h"
#include "sim.h"

static SimState *global_state = NULL;

int main(int argc, char** argv) {

    // defaults
    static SimState state = {
        .mouseX = -1,
        .mouseY = -1,
        .mouseClicked = false,
        .amplitude = 10.0f,
        .boxSize = 50.0f
    };

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(SIZE_X, SIZE_Y);

    int main_window = glutCreateWindow("CUDA EM Sim");

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
        exit(1);
    }

    init_gpu(global_state);

    GLUI *glui = GLUI_Master.create_glui("Controls");

    glui->add_spinner("Amplitude", GLUI_SPINNER_FLOAT, &(global_state->amplitude), 1, control_cb)->set_float_limits(2.0, 50.0);
    glui->add_spinner("Box_Size", GLUI_SPINNER_FLOAT, &(global_state->boxSize), 1, control_cb)->set_float_limits(20.0, 200.0);
    glui->set_main_gfx_window(main_window);

    create_pbo(global_state);

    glutDisplayFunc(display);
    set_sim_state(global_state); //register state with input callbacks
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse_func);

    glutMainLoop();
    return 0;
}