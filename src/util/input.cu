#include "input.cuh"

static SimState* local_state = NULL;

void set_sim_state(SimState* state) {
    local_state = state;
}

void keyboard(unsigned char key, int x, int y) {
    if (key == 27) { //eSC key
        cleanup(local_state);
        exit(0);
    }
}
void mouse_func(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            local_state->mouseClicked = true;
        } else {
            local_state->mouseClicked =false;
        }
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        dim3 block(16, 16);
        dim3 grid((SIZE_X + 15) / 16, (SIZE_Y + 15) / 16);
        add_box<<<grid, block>>>(local_state->d_Pec_Mask, x, SIZE_Y - y,local_state->boxSize, SIZE_X, SIZE_Y);
    }
}

void motion_func(int x, int y) {
    local_state->mouseX = x;
    local_state->mouseY = y;
}
