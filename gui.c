#include <GL/glut.h>
#include <vector.h>
#include <cmath>

// Window dimensions
const int WIDTH = 800;
const int HEIGHT = 600;

// Shape types
enum ShapeType { POINT, LINE, RECTANGLE, CIRCLE, POLYGON };

// Color options
enum ColorOption { RED, GREEN, BLUE, WHITE, BLACK };

// Point structure
struct Point2D {
    int x, y;
    Point2D(int _x = 0, int _y = 0) : x(_x), y(_y) {}
};

// Shape structure
struct Shape {
    ShapeType type;
    std::vector<Point2D> points;
    float color[3];
};

// Global variables
std::vector<Shape> shapes;
Shape currentShape;
bool isDrawing = false;
bool polygonComplete = false;

// Initialize OpenGL
void init() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, WIDTH, 0.0, HEIGHT);
    glPointSize(5.0f);
    glLineWidth(2.0f);
    
    // Set default color (white)
    currentShape.color[0] = 1.0f;
    currentShape.color[1] = 1.0f;
    currentShape.color[2] = 1.0f;
    currentShape.type = LINE;
}

// Set color based on selection
void setColor(int option) {
    switch(option) {
        case RED:
            currentShape.color[0] = 1.0f;
            currentShape.color[1] = 0.0f;
            currentShape.color[2] = 0.0f;
            break;
        case GREEN:
            currentShape.color[0] = 0.0f;
            currentShape.color[1] = 1.0f;
            currentShape.color[2] = 0.0f;
            break;
        case BLUE:
            currentShape.color[0] = 0.0f;
            currentShape.color[1] = 0.0f;
            currentShape.color[2] = 1.0f;
            break;
        case WHITE:
            currentShape.color[0] = 1.0f;
            currentShape.color[1] = 1.0f;
            currentShape.color[2] = 1.0f;
            break;
        case BLACK:
            currentShape.color[0] = 0.0f;
            currentShape.color[1] = 0.0f;
            currentShape.color[2] = 0.0f;
            break;
    }
}

// Set shape type based on selection
void setShapeType(int type) {
    currentShape.type = static_cast<ShapeType>(type);
    polygonComplete = false;
}

// Clear all shapes
void clearScreen(int option) {
    shapes.clear();
    glutPostRedisplay();
}

// Create right-click menu
void createMenu() {
    int shapeMenu = glutCreateMenu(setShapeType);
    glutAddMenuEntry("Point", POINT);
    glutAddMenuEntry("Line", LINE);
    glutAddMenuEntry("Rectangle", RECTANGLE);
    glutAddMenuEntry("Circle", CIRCLE);
    glutAddMenuEntry("Polygon", POLYGON);
    
    int colorMenu = glutCreateMenu(setColor);
    glutAddMenuEntry("Red", RED);
    glutAddMenuEntry("Green", GREEN);
    glutAddMenuEntry("Blue", BLUE);
    glutAddMenuEntry("White", WHITE);
    glutAddMenuEntry("Black", BLACK);
    
    int mainMenu = glutCreateMenu(clearScreen);
    glutAddSubMenu("Shapes", shapeMenu);
    glutAddSubMenu("Colors", colorMenu);
    glutAddMenuEntry("Clear All", 0);
    
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

// Draw a circle using midpoint algorithm
void drawCircle(Point2D center, int radius) {
    int x = radius;
    int y = 0;
    int err = 0;

    while (x >= y) {
        glVertex2i(center.x + x, center.y + y);
        glVertex2i(center.x + y, center.y + x);
        glVertex2i(center.x - y, center.y + x);
        glVertex2i(center.x - x, center.y + y);
        glVertex2i(center.x - x, center.y - y);
        glVertex2i(center.x - y, center.y - x);
        glVertex2i(center.x + y, center.y - x);
        glVertex2i(center.x + x, center.y - y);

        if (err <= 0) {
            y += 1;
            err += 2*y + 1;
        }
        if (err > 0) {
            x -= 1;
            err -= 2*x + 1;
        }
    }
}

// Display callback
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw all stored shapes
    for (const auto& shape : shapes) {
        glColor3fv(shape.color);
        switch(shape.type) {
            case POINT:
                glBegin(GL_POINTS);
                glVertex2i(shape.points[0].x, shape.points[0].y);
                glEnd();
                break;
                
            case LINE:
                glBegin(GL_LINES);
                glVertex2i(shape.points[0].x, shape.points[0].y);
                glVertex2i(shape.points[1].x, shape.points[1].y);
                glEnd();
                break;
                
            case RECTANGLE:
                glBegin(GL_LINE_LOOP);
                glVertex2i(shape.points[0].x, shape.points[0].y);
                glVertex2i(shape.points[1].x, shape.points[0].y);
                glVertex2i(shape.points[1].x, shape.points[1].y);
                glVertex2i(shape.points[0].x, shape.points[1].y);
                glEnd();
                break;
                
            case CIRCLE:
                glBegin(GL_POINTS);
                drawCircle(shape.points[0], 
                    static_cast<int>(sqrt(
                        pow(shape.points[1].x - shape.points[0].x, 2) + 
                        pow(shape.points[1].y - shape.points[0].y, 2)
                    ));
                glEnd();
                break;
                
            case POLYGON:
                glBegin(GL_LINE_LOOP);
                for (const auto& p : shape.points) {
                    glVertex2i(p.x, p.y);
                }
                glEnd();
                break;
        }
    }
    
    // Draw current shape being created
    if (isDrawing && !currentShape.points.empty()) {
        glColor3fv(currentShape.color);
        
        switch(currentShape.type) {
            case POINT:
                glBegin(GL_POINTS);
                glVertex2i(currentShape.points[0].x, currentShape.points[0].y);
                glEnd();
                break;
                
            case LINE:
                glBegin(GL_LINES);
                glVertex2i(currentShape.points[0].x, currentShape.points[0].y);
                glVertex2i(currentShape.points.back().x, currentShape.points.back().y);
                glEnd();
                break;
                
            case RECTANGLE:
                if (currentShape.points.size() > 1) {
                    glBegin(GL_LINE_LOOP);
                    glVertex2i(currentShape.points[0].x, currentShape.points[0].y);
                    glVertex2i(currentShape.points.back().x, currentShape.points[0].y);
                    glVertex2i(currentShape.points.back().x, currentShape.points.back().y);
                    glVertex2i(currentShape.points[0].x, currentShape.points.back().y);
                    glEnd();
                }
                break;
                
            case CIRCLE:
                if (currentShape.points.size() > 1) {
                    glBegin(GL_POINTS);
                    drawCircle(currentShape.points[0], 
                        static_cast<int>(sqrt(
                            pow(currentShape.points.back().x - currentShape.points[0].x, 2) + 
                            pow(currentShape.points.back().y - currentShape.points[0].y, 2)
                        )));
                    glEnd();
                }
                break;
                
            case POLYGON:
                glBegin(GL_LINE_STRIP);
                for (const auto& p : currentShape.points) {
                    glVertex2i(p.x, p.y);
                }
                if (!polygonComplete) {
                    glVertex2i(currentShape.points.back().x, currentShape.points.back().y);
                }
                glEnd();
                break;
        }
    }
    
    glutSwapBuffers();
}

// Mouse callback
void mouse(int button, int state, int x, int y) {
    y = HEIGHT - y; // Flip y coordinate
    
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            isDrawing = true;
            
            if (currentShape.type == POLYGON && !polygonComplete) {
                // Add point to existing polygon
                currentShape.points.push_back(Point2D(x, y));
            } else {
                // Start new shape
                currentShape.points.clear();
                currentShape.points.push_back(Point2D(x, y));
                polygonComplete = false;
            }
        }
        else if (state == GLUT_UP) {
            if (currentShape.type == POLYGON) {
                // For polygon, don't complete until right-click
            } else {
                // For other shapes, complete on mouse up
                isDrawing = false;
                shapes.push_back(currentShape);
                currentShape.points.clear();
            }
            glutPostRedisplay();
        }
    }
    else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        if (currentShape.type == POLYGON && !polygonComplete && currentShape.points.size() > 2) {
            // Complete the polygon
            isDrawing = false;
            polygonComplete = true;
            shapes.push_back(currentShape);
            currentShape.points.clear();
            glutPostRedisplay();
        }
    }
}

// Motion callback for dragging
void motion(int x, int y) {
    y = HEIGHT - y;
    
    if (isDrawing) {
        if (currentShape.type == POLYGON) {
            // For polygon, points are added on click, not drag
            return;
        }
        
        if (currentShape.points.size() < 2) {
            currentShape.points.push_back(Point2D(x, y));
        } else {
            currentShape.points.back() = Point2D(x, y);
        }
        glutPostRedisplay();
    }
}

// Keyboard callback
void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case 27: // ESC key
            exit(0);
            break;
        case 'c':
            clearScreen(0);
            break;
    }
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("OpenGL Drawing Tool");
    
    init();
    createMenu();
    
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    
    glutMainLoop();
    return 0;
}