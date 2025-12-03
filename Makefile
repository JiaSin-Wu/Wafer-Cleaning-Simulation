# Compilers
CXX = g++
CC = gcc
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = -std=c++17 -O3 -arch=sm_70 -Xptxas=-v #--use_fast_math


# Targets
TARGET_CUDA = final_gpu
TARGET_CPU = final_cpu

# CUDA version
all: $(TARGET_CUDA)

$(TARGET_CUDA): final_gpu.cu $(LODEPNG)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) final_gpu.cu $(LODEPNG) -o $(TARGET_CUDA)

# CPU version
cpu: $(TARGET_CPU)

$(TARGET_CPU): final_cpu
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) final_cpu.cpp $(LODEPNG) -o $(TARGET_CPU)

# Clean
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA)
