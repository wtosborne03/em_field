NVCC = nvcc
CUDA_ARCH = sm_35

# Flags
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3
INCLUDES = -I./src -I./src/util -I./libs/glui/include
LDFLAGS = -L./libs/glui/lib
LIBS = -lglui -lglut -lGL -lGLU -lGLEW

TARGET = em_vis

#all .cu and .c source files
SRCS = $(wildcard src/*.cu) $(wildcard src/util/*.cu) $(wildcard src/util/*.c)
OBJS = $(patsubst %.cu,%.o,$(patsubst %.c,%.o,$(SRCS)))

#all .h and .cuh header files
HEADERS = $(wildcard src/*.h) $(wildcard src/util/*.h) $(wildcard src/*.cuh) $(wildcard src/util/*.cuh) $(wildcard libs/glui/include/GL/*.h)

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