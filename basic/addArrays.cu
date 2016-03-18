// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2015/2016, March 15 - April 15
// ###
// ###

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__device__ float add(float a, float b) {
  return a + b;
}

__global__ void add_global(float *d_a, float *d_b, float *d_c, int n) {
  int ind = threadIdx.x + blockDim.x * blockIdx.x;
  if (ind < n) d_c[ind] = add(d_a[ind], d_b[ind]);
}

int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;



    // GPU computation
    // ###
    // ### TODO: Implement the array addition on the GPU, store the result in "c"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "helper.h"

    // allocate GPU memory
    size_t nbytes = n*sizeof(float);
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    cudaMalloc(&d_a, nbytes);
    cudaMemset(d_a, 0, nbytes);
    cudaMalloc(&d_b, nbytes);
    cudaMemset(d_b, 0, nbytes);
    cudaMalloc(&d_c, nbytes);
    cudaMemset(d_c, 0, nbytes);

    //copy host memory to device
    cudaMemcpy( d_a, a, nbytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, b, nbytes, cudaMemcpyHostToDevice );

    // launch kernel
    dim3 block = dim3(32,1,1);
    dim3 grid = dim3(1,1,1);
    add_global <<<grid,block>>> (d_a, d_b, d_c, n);
    CUDA_CHECK;

    // copy device memory to host
    cudaMemcpy( c, d_c, nbytes, cudaMemcpyDeviceToHost );

    // free GPU arrays
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;
}
