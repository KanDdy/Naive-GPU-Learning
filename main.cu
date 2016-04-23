#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "timer.h"

__device__ float theta(float x) {
	return 1 / (1 + expf(-x));
}

__device__ float inner_product(float *x1, float *x2, int len) {
	float sum = 0;

	for(int i = 0; i < len; i++) {
		sum += x1[i] * x2[i];
	}

	return sum;
}

__global__ void kernel_gradient(float *X, float *y, float *err, int row, int col, float *w) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= row) {
		return;
	}

	float dot = inner_product(X + id * col, w, col);
	float th = theta(dot * (-y[id]));

	for(int i = 0; i < col; i++) {
		err[id * col + i] = (-y[id]) * X[id * col + i] * th;
	}

	return;
}

__global__ void kernel_reduce(float * d_out, float * d_in, int row, int col) {
	extern __shared__ float sd[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	for(int i = 0; i < col; i++) {
		sd[tid * col + i] = d_in[x * col + i];
	}

	__syncthreads();

	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if(tid < s && (x + s) < row) {
			for(int i = 0; i < col; i++) {
				sd[tid * col + i] += sd[(tid + s) * col + i];
			}
		}

		__syncthreads();
	}

	if(tid == 0) {
		for(int i = 0; i < col; i++) {
			atomicAdd(&d_out[i], sd[i] / row);
		}
	}
}

int scan(int * array, int ARRAY_SIZE) {
	const int grid_x = 256;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	int * h_in = array, h_out[1];
	int * d_in, * d_out;

	int block_num = ARRAY_SIZE / grid_x + (ARRAY_SIZE % grid_x == 0 ? 0 : 1);

	const dim3 block_size(block_num, 1, 1);
	const dim3 grid_size(grid_x, 1, 1);

	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, sizeof(int));
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemset(d_out, 0, sizeof(int));

	//kernel_reduce<<<block_size, grid_size, sizeof(int) * grid_x>>>(d_out, d_in, ARRAY_SIZE);

	cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	return h_out[0];
}

void logistic(float *X, float *y, int row, int col, int iteration, float eta) {
	const int grid_x = 512;
	const int X_SIZE = row * col * sizeof(float);
	const int y_SIZE = row * sizeof(float);
	const int param_SIZE = col * sizeof(float);

	float *h_X = X, *d_X;
	float h_w[col], *d_w, h_delta[col], *d_delta;
	float *d_err;
	float *h_y = y, *d_y;

	for(int i = 0; i < col; i++) {
		h_w[i] = 1;
		h_delta[i] = 0;
	}

	cudaMalloc((void**) &d_X, X_SIZE);
	cudaMemcpy(d_X, h_X, X_SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_y, y_SIZE);
	cudaMemcpy(d_y, h_y, y_SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_err, X_SIZE);

	cudaMalloc((void**) &d_w, param_SIZE);
	cudaMemset(d_w, 1, param_SIZE);

	cudaMalloc((void**) &d_delta, param_SIZE);
	//cudaMemset(d_delta, 0, param_SIZE);

	int block_num = row / grid_x + (row % grid_x == 0 ? 0 : 1);

	const dim3 block_size(block_num, 1, 1);
	const dim3 grid_size(grid_x, 1, 1);

	for(int i = 0; i < iteration; i++) {
		printf("Iteration:%d\n", i);

		cudaMemcpy(d_w, h_w, param_SIZE, cudaMemcpyHostToDevice);
		cudaMemset(d_delta, 0, param_SIZE);

		kernel_gradient<<<block_size, grid_size>>>(d_X, d_y, d_err, row, col, d_w);
		kernel_reduce<<<block_size, grid_size, sizeof(float) * grid_x * col>>>(d_delta, d_err, row, col);
		cudaMemcpy(h_delta, d_delta, param_SIZE, cudaMemcpyDeviceToHost);

		printf("Delta\tW:\n");
		for(int k = 0; k < col; k++) {
			printf("%f\t%f\n", h_delta[k], h_w[k]);
		}
		printf("\n\n");

		for(int k = 0; k < col; k++) {
			h_w[k] -= eta * h_delta[k];
		}
	}

	cudaFree(d_X);
	cudaFree(d_y);
	cudaFree(d_err);
	cudaFree(d_w);
	cudaFree(d_delta);
}

int main() {
	GpuTimer timer;
	const int ROW = 1000000, COL = 18;
	const int ITERATION = 100;
	float *X = new float[ROW * COL];
	float *y = new float[ROW * COL];;

	FILE *fp;
	if(fp = fopen("SUSY.csv", "r")) {
		printf("Open file success!\n");
	} else {
		printf("Open file falied!\n");
		return 1;
	}

	for(int i = 0; i < ROW; i++) {
		fscanf(fp, "%f", y + i);
		for(int j = 0; j < COL; j++) {
			fscanf(fp, "%f", X + i*COL + j);
		}
	}

	fclose(fp);

	timer.Start();
	logistic(X, y, ROW, COL, ITERATION, 0.1);
	timer.Stop();

	//printf("gpu:[%d] - true:[%d]\n", result, sum);
	printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	delete[] X;
	delete[] y;

	return 0;
}
