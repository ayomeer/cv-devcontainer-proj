// Cuda libraries automatically included at compile-time by magic of Cuda toolkit

#include <iostream>

// Kernel function to run on each thread
__global__ void vectorAdd(int* a, int* b, int* c) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main(void) {
	
	int a[] = {1,2,3,4,5,6};
	int b[] = {2,3,4,5,6,7};
	auto NUMBER_OF_VECTORS = sizeof(a) / sizeof(int);
	int c[NUMBER_OF_VECTORS] = {0};

	// create pointers into the GPU
	int* cudaA;
	int* cudaB;
	int* cudaC;

	// allocate memory in the GPU
	cudaMalloc(&cudaA, sizeof(a));
	cudaMalloc(&cudaB, sizeof(b));
	cudaMalloc(&cudaC, sizeof(c));

	// copy into GPU
	cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, sizeof(a), cudaMemcpyHostToDevice);

	auto GRID_SIZE = 1; 				 	// number of blocks in grid
	auto BLOCK_SIZE = NUMBER_OF_VECTORS; 	// size of elements in block

	vectorAdd <<< GRID_SIZE, BLOCK_SIZE >>> (cudaA, cudaB, cudaC);

	// copy back out of GPU
	cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

	for (int i = 0; i < NUMBER_OF_VECTORS; i++) {
		std::cout << c[i] << " ";
    }
    std::cout << std::endl;

	return 0;
}

