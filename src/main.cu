#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <GL/glui.h>

#include "util/input.cuh"
#include "sim.cuh"

static SimState *global_state = NULL;
static SimState state;

#define NUM_MATERIALS 10

void display_wrapper()
{
    if (global_state)
    {
        display(global_state);
    }
}

int main(int argc, char **argv)
{
    
    material material_list[NUM_MATERIALS] = {
        {8.854e-12, 1.2566e-6, 0.0, 0.0f, 0.0f, 0.0f, "Vacuum"},        // VACUUM
        {1.0006 * 8.854e-12, 1.2566e-6, 0.0, 0.5f, 0.5f, 1.0f, "Air"},  // AIR
        {4.5 * 8.854e-12, 1.2566e-6, 1e-12, 0.3f, 0.8f, 1.0f, "Glass"}, // GLASS
        {8.854e-12, 1.2566e-6, 1e7, 0.8f, 0.8f, 0.8f, "Metal"},        // METAL
        {2.2 * 8.854e-12, 1.2566e-6, 1e-14,  1.0f, 0.8f, 0.6f,  "Teflon"},        // Low-loss microwave dielectric
        {10.0 * 8.854e-12, 1.2566e-6, 1e-4,  1.0f, 0.4f, 0.6f,  "Ceramic"},       // High-permittivity material
        {1.5 * 8.854e-12, 200 * 1.2566e-6, 1e-6, 0.2f, 0.9f, 0.2f, "Ferrite"},     // Magnetic material
        {2.5 * 8.854e-12, 1.2566e-6, 5.0e4,  1.0f, 1.0f, 0.4f,  "Salt Water"},    // High-conductivity, dispersive behavior
        {6.0 * 8.854e-12, 1.2566e-6, 1.0,    0.9f, 0.6f, 1.0f,  "Human Tissue"},
{-0.254e-12, -0.1566e-6, 1e4, 0.0f, 1.0f, 1.0f, "Metamaterial"}
    };

    

    // defaults
    state = (SimState){
        .mouseX = -1,
        .mouseY = -1,
        .mouseClicked = false,
        .shapeClicked = false,
        .amplitude = 10.0f,
        .boxSize = 50.0f,
        .dx = 0.5,
        .materials = material_list,
        .selected_material = 0,
        .num_materials = NUM_MATERIALS,
        .shape_type = 0
        };
    global_state = &state;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(SIZE_X, SIZE_Y);

    int main_window = glutCreateWindow("CUDA EM Sim");

    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
        exit(1);
    }

    init_gpu(global_state);

    GLUI *glui = GLUI_Master.create_glui("Controls");

    glui->add_spinner("Simulation Step Speed (dx)", GLUI_SPINNER_FLOAT, &(global_state->dx), 1, control_cb)->set_float_limits(0.05, 1.0);
    glui->add_spinner("Pulse Amplitude", GLUI_SPINNER_FLOAT, &(global_state->amplitude), 1, control_cb)->set_float_limits(1.0, 100.0);
    glui->add_spinner("Pulse Spread", GLUI_SPINNER_FLOAT, &(global_state->spread), 1, control_cb)->set_float_limits(1.0, 100.0);

    glui->add_spinner("Box Size", GLUI_SPINNER_FLOAT, &(global_state->boxSize), 1, control_cb)->set_float_limits(5.0, 200.0);

    GLUI_Listbox *material_listbox = glui->add_listbox("Material Type", &(global_state->selected_material));
    for (int i = 0; i < global_state->num_materials; i++)
    {
        material_listbox->add_item(i, global_state->materials[i].name);
    }

    GLUI_Listbox *shape_type_listbox = glui->add_listbox("Shape Type", &(global_state->shape_type));
    shape_type_listbox->add_item(0, "Square");
    shape_type_listbox->add_item(1, "Circle");


    glui->set_main_gfx_window(main_window);

    create_pbo(global_state);

    glutDisplayFunc(display_wrapper);
    set_sim_state(global_state); // register state with input callbacks
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);

    glutMainLoop();
    return 0;
}