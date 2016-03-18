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
#include <chrono> // measure times
using namespace std;
using namespace std::chrono;



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


__device__ float square(float a) {
  return a*a;
}

__global__ void square_global(float *d_a, int n) {
  int ind = threadIdx.x + blockDim.x * blockIdx.x;
  if (ind < n) d_a[ind] = square(d_a[ind]);
}

int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
    for(int i=0; i<n; i++) a[i] = i;

    // timing
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }
    //timing
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    // print timing
    cout << "CPU time: "<< duration << "ms" << endl;
    cout << endl;



    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) a[i] = i;


    // ###
    // ### TODO: Implement the "square array" operation on the GPU and store the result in "a"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "helper.h"

    // timing START
    high_resolution_clock::time_point t3 = high_resolution_clock::now();

    // allocate the GPU memory
    size_t nbytes = n*sizeof(float);
    float *d_a = NULL;
    cudaMalloc(&d_a, nbytes);
    cudaMemset(d_a, 0, nbytes);

    //copy host memory to device
    cudaMemcpy( d_a, a, nbytes, cudaMemcpyHostToDevice );

    // launch kernel
    dim3 block = dim3(32,1,1);
    dim3 grid = dim3(1,1,1);
    square_global <<<grid,block>>> (d_a, n);
    CUDA_CHECK;

    // copy device memory to host
    cudaMemcpy( a, d_a, nbytes, cudaMemcpyDeviceToHost );

    //timing END
    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>( t4 - t3 ).count();

    // free memory
    cudaFree(d_a);


    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    // print timing
    cout << "GPU time: "<< duration2 << "ms" << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
}
