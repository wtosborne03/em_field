NVCC = nvcc
CUDA_ARCH = sm_35

# Flags
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3
INCLUDES = -I./src -I./src/util -I./libs/glui/include
LDFLAGS = -L./libs/glui/lib
LIBS = -lglui -lglut -lGL -lGLU -lGLEW

TARGET = em_vis

SRCS = $(wildcard src/*.cu src/*.c src/util/*.c)
OBJS = $(patsubst %.cu,%.o,$(patsubst %.c,%.o,$(SRCS)))

HEADERS = $(wildcard src/*.h src/util/*.h src/*.cuh) common.h libs/glui/include/GL/glui.h

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)
	@echo "Build complete. Run ./$(TARGET)"

%.o: %.cu $(HEADERS)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c $(HEADERS)
	@echo "Compiling C $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x c -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(TARGET)
	@echo "Cleanup complete."

.PHONY: all clean