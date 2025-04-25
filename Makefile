CUDA_ARCH=sm_35
INCLUDES=-I./libs/glui/include
LIBS=-L./libs/glui/lib -lglui -lglut -lGL -lGLU -lGLEW -L/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-11.8.0-in72fn46ydgmi5ak67tvzjll5dz4w43u/lib64 -lcudart

all: em_vis

main.o: src/main.cu
	@echo "Compiling main.cu..."
	nvcc -arch=$(CUDA_ARCH) $(INCLUDES) -c src/main.cu -o main.o || exit 1

em_vis: main.o
	@echo "Linking em_vis..."
	g++ main.o $(LIBS) -o em_vis || exit 1
	@echo "Build complete. Run ./em_vis"

clean:
	rm -f *.o em_vis