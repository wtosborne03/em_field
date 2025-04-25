NVCC = nvcc
CUDA_ARCH = sm_35 # Update this for your GPU (e.g., sm_75, sm_86)

# Flags
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 # Add -g for debugging
INCLUDES = -I./src -I./src/util -I./libs/glui/include
LDFLAGS = -L./libs/glui/lib
LIBS = -lglui -lglut -lGL -lGLU -lGLEW

TARGET = em_vis

# Source and Object files
SRCS = $(wildcard src/*.cu src/*.c src/util/*.c)
OBJS = $(patsubst %.cu,%.o,$(patsubst %.c,%.o,$(SRCS)))

# Header files (simplified dependency tracking)
HEADERS = $(wildcard src/*.h src/util/*.h src/*.cuh) common.h libs/glui/include/GL/glui.h

# Default target
all: $(TARGET)

# Linking rule
$(TARGET): $(OBJS)
    @echo "Linking $(TARGET)..."
    $(NVCC) $(NVCCFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)
    @echo "Build complete. Run ./$(TARGET)"

%.o: %.cu $(HEADERS)
    @echo "Compiling $<..."
    $(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c $(HEADERS)
    @echo "Compiling $<..."
    $(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
    @echo "Cleaning up..."
    rm -f $(OBJS) $(TARGET)
    @echo "Cleanup complete."

.PHONY: all clean