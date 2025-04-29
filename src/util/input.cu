#include "input.cuh"

static SimState *local_state = NULL;

void set_sim_state(SimState *state)
{
    local_state = state;
}

void keyboard(unsigned char key, int x, int y)
{
    if (key == 27)
    {
        cleanup(local_state);
        exit(0);
    }
}
void mouse_func(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN)
        {
            local_state->mouseX = x;
            local_state->mouseY = y;
            local_state->mouseClicked = true;
        }
        else
        {
            local_state->mouseClicked = false;
        }
    }
    if (button == GLUT_RIGHT_BUTTON)
    {
        if (state == GLUT_DOWN) {
            local_state->mouseX = x;
            local_state->mouseY = y;
            local_state->shapeClicked = true;
        }
        else {
            local_state->shapeClicked = false;
        }


    }
}

void motion_func(int x, int y)
{
    local_state->mouseX = x;
    local_state->mouseY = y;
}

void control_cb(int id)
{
    // This function is called when a GLUI control changes
}