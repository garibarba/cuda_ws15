// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2015/2016, March 15 - April 15
// ###
// ###

#include "helper.h"
#include <iostream>
using namespace std;

__global__ void block_sum(float *input, float *results, size_t n) {
  extern __shared__ float sdata[];
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int tx = threadIdx.x;
  // load input into __shared__ memory
  if (i < n) {
    sdata[tx] = input[i];
    __syncthreads();
  } else {
    sdata[tx] = 0;
  }
  if (i < n) {
    // block-wide reduction in __shared__ mem
    for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if(tx < offset) {
        // add a partial sum upstream to our own
        sdata[tx] += sdata[tx + offset];
      }
      __syncthreads();
    }
    // finally, thread 0 writes the result
    if(threadIdx.x == 0) {
      // note that the result is per-block
      // not per-thread
      results[blockIdx.x] = sdata[0];
    }
  }
}

int main(int argc, char **argv)
{
    int n = 10000;
    // alloc and init input array on host (CPU)
    float *a = new float[n];
    for (int i = 0; i < n; i++) {
      a[i] = 1;
    }
    // parameters
    int blocklength = 1024;
    int nblocks = (n + blocklength -1)/blocklength;
    size_t nbytes = n*sizeof(float);
    // other variables
    float *aux, *results;
    results = new float[nblocks];
    // alloc device arrays

    Timer timer; timer.start();

    float *d_a = NULL;
    float *d_results = NULL;
    cudaMalloc(&d_a, nbytes);
    cudaMalloc(&d_results, nblocks*sizeof(float));
    cudaMemcpy( d_a, a, nbytes, cudaMemcpyHostToDevice );

    dim3 block = dim3(blocklength,1,1);
    dim3 grid = dim3(nblocks, 1, 1 );
    // only one reduction
    // block_sum <<<grid,block,blocklength*sizeof(float)>>> (d_a, d_results, n);
    // reductions until size 1
    while (true) {
      block_sum <<<grid,block,blocklength*sizeof(float)>>> (d_a, d_results, n);
      if (nblocks == 1) break;
      cudaMemcpy( d_a, d_results, nblocks*sizeof(float), cudaMemcpyDeviceToDevice );
      n = nblocks;
      nblocks = (n + blocklength -1)/blocklength;
      grid = dim3(nblocks, 1, 1 );
    }

    // show results
    // float total = 0;
    cudaMemcpy( results, d_results, sizeof(float), cudaMemcpyDeviceToHost );

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "My time: " << t*1000 << " ms" << endl;

    // for (int i = 0; i < n; i++) {
    //   total += results[i];
    //   cout << results[i] << endl;
    // }
    // cout << "total :" << total << endl;
    cout << "total :" << results[0] << endl;


}
