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
#include <stdio.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA

__device__ size_t clamp(size_t ind, size_t minval, size_t maxval) {
  return min(max(minval, ind), maxval);
}
__global__ void gradient(float *d_imgIn, float *d_imgGrad_x, float *d_imgGrad_y, int w, int h, int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + w * y;
  for (int c = 0; c < nc; c++) {
    if (x + 1 < w) d_imgGrad_x[ind + (size_t)c*w*h] = d_imgIn[ind + (size_t)c*w*h + 1] - d_imgIn[ind + (size_t)c*w*h]; // derivative along x
    else           d_imgGrad_x[ind + (size_t)c*w*h] = 0;
    if (y + 1 < h) d_imgGrad_y[ind + (size_t)c*w*h] = d_imgIn[ind + (size_t)c*w*h + w] - d_imgIn[ind + (size_t)c*w*h]; // derivative along y
    else           d_imgGrad_y[ind + (size_t)c*w*h] = 0;
  }
}

__global__ void ri_gradient(float *d_imgIn, float *d_imgGrad_x, float *d_imgGrad_y, int w, int h, int nc) {
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;
  for (int c = 0; c < nc; c++) {
    if (x < w && y < h) {
      d_imgGrad_x[x + y*w + (size_t)c*w*h] =
                      ( 3*d_imgIn[clamp(x+1,0,w-1) + w*clamp(y-1,0,h-1) + (size_t)c*w*h]
                        + 10*d_imgIn[clamp(x+1,0,w-1) + w*clamp(y,0,h-1) + (size_t)c*w*h]
                        + 3*d_imgIn[clamp(x+1,0,w-1) + w*clamp(y+1,0,h-1) + (size_t)c*w*h]
                        - 3*d_imgIn[clamp(x-1,0,w-1) + w*clamp(y-1,0,h-1) + (size_t)c*w*h]
                        - 10*d_imgIn[clamp(x-1,0,w-1) + w*clamp(y,0,h-1) + (size_t)c*w*h]
                        - 3*d_imgIn[clamp(x-1,0,w-1) + w*clamp(y+1,0,h-1) + (size_t)c*w*h] ) / 32.f; // derivative along x
      d_imgGrad_y[x + y*w + (size_t)c*w*h] =
                      ( 3*d_imgIn[clamp(x-1,0,w-1) + w*clamp(y+1,0,h-1) + (size_t)c*w*h]
                        + 10*d_imgIn[clamp(x,0,w-1) + w*clamp(y+1,0,h-1) + (size_t)c*w*h]
                        + 3*d_imgIn[clamp(x+1,0,w-1) + w*clamp(y+1,0,h-1) + (size_t)c*w*h]
                        - 3*d_imgIn[clamp(x-1,0,w-1) + w*clamp(y-1,0,h-1) + (size_t)c*w*h]
                        - 10*d_imgIn[clamp(x,0,w-1) + w*clamp(y-1,0,h-1) + (size_t)c*w*h]
                        - 3*d_imgIn[clamp(x+1,0,w-1) + w*clamp(y-1,0,h-1) + (size_t)c*w*h] ) / 32.f; // derivative along x
      // d_imgGrad[x + y*w + (size_t)c*w*h + w*h*nc] = d_imgIn[ind + (size_t)c*w*h + w] - d_imgIn[ind + (size_t)c*w*h]; // derivative along y
    }
  }
}

__global__ void divergence_2d(float *d_imgV1, float *d_imgV2, float *d_div, int w, int h, int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  float dx;
  float dy;
  size_t ind = x + w * y;
  for (int c = 0; c < nc; c++) {
    if (x > 0) dx = d_imgV1[ind + (size_t)c*w*h] - d_imgV1[ind + (size_t)c*w*h - 1]; // derivative along x
    else       dx = d_imgV1[ind + (size_t)c*w*h];
    if (y > 0) dy = d_imgV2[ind + (size_t)c*w*h] - d_imgV2[ind + (size_t)c*w*h - w]; // derivative along y
    else       dy = d_imgV2[ind + (size_t)c*w*h];
    d_div[ind + (size_t)c*w*h] = dx + dy;
  }
}

__global__ void l2norm(float *d_imgIn, float *d_imgOut, int w, int h, int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + w * y;
  for (int c = 0; c < nc; c++) {
    d_imgOut[ind] += powf(d_imgIn[ind + (size_t)c*w*h], 2);
  }
  d_imgOut[ind] = sqrtf(d_imgOut[ind]);
}

__host__ __device__ float g_one(float s = 1, float epsilon = 1) {
  return 1.f;
}

__host__ __device__ float g_max(float s, float epsilon) {
  return 1.f/max(s, epsilon);
}

__host__ __device__ float g_exp(float s, float epsilon) {
  return expf(-s*s/ epsilon)/ epsilon;
}

// calculate the norm of a 2D vector
__host__ __device__ float norm_2d(float v1, float v2) {
  return sqrtf(v1*v1 + v2*v2);
}

__global__ void scalar_mult(float *imgV1, float *imgV2, int w, int h, int nc, float epsilon = 1) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x < w && y < h) {
    int idx;
    float norma;
    // for every channel
    for (int c = 0; c < nc; c++) {
      idx = x + y*w + c*w*h;
      norma = norm_2d(imgV1[idx], imgV2[idx]);
      // imgV1[idx] *= g_one(norma, epsilon);
      // imgV2[idx] *= g_one(norma, epsilon);
      imgV1[idx] *= g_max(norma, epsilon);
      imgV2[idx] *= g_max(norma, epsilon);
      // imgV1[idx] *= g_exp(norma, epsilon);
      // imgV2[idx] *= g_exp(norma, epsilon);
    }
  }
}

__global__ void time_step(float *imgIn, float *imgGrad, float tau, int w, int h, int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x < w && y < h) {
    int idx;
    // for every channel
    for (int c = 0; c < nc; c++) {
      idx = x + y*w + c*w*h;
      imgIn[idx] += tau*imgGrad[idx];
    }
  }
}









int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif

    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;

    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed

    // number of time steps
    int N = 10;
    getParam("N", N, argc, argv);
    cout << "N: " << N << endl;

    // size of time steps
    float tau = 0.25;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    cout << "tau x N = " << tau*N << ", if anisotropic is equivalent to sigma = " << sqrt(2*tau*N) << endl;

    // g function parameter epsilon
    float epsilon = 0.01;
    getParam("epsilon", epsilon, argc, argv);
    cout << "epsilon: " << epsilon << endl;



    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;

#else

    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }

#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    // cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);






    cout << "GPU Laplacian absolute value" << endl;
    Timer timer; timer.start();
    // ###
    // ###

    float *d_imgIn = NULL;
    float *d_imgTimeGrad = NULL;
    float *d_imgGrad_x = NULL;
    float *d_imgGrad_y = NULL;
    float *d_imgOut = NULL;
    cudaMalloc( &d_imgIn, w*h*nc*sizeof(float) );
    cudaMalloc( &d_imgTimeGrad, w*h*nc*sizeof(float) );
    cudaMalloc( &d_imgGrad_x, w*h*nc*sizeof(float) );
    cudaMalloc( &d_imgGrad_y, w*h*nc*sizeof(float) );
    cudaMalloc( &d_imgOut, nc*w*h*sizeof(float) );
    cudaMemset( d_imgOut, 0, nc*w*h*sizeof(float) );

    //copy host memory to device
    cudaMemcpy( d_imgIn, imgIn, w*h*nc*sizeof(float), cudaMemcpyHostToDevice );

    // kernel only timer
    Timer timer2; timer2.start();
    // launch kernel
    dim3 block = dim3(32,8,1);
    dim3 grid = dim3( (w + block.x -1)/block.x, (h + block.y -1)/block.y, 1);

    for (int n = 0; n < N; n++) {
      gradient <<<grid,block>>> (d_imgIn, d_imgGrad_x, d_imgGrad_y, w, h, nc); CUDA_CHECK;
      scalar_mult <<<grid,block>>> (d_imgGrad_x, d_imgGrad_y, w, h, nc, epsilon); CUDA_CHECK;
      divergence_2d <<<grid,block>>> (d_imgGrad_x, d_imgGrad_y, d_imgTimeGrad, w, h, nc); CUDA_CHECK;
      time_step <<<grid,block>>> (d_imgIn, d_imgTimeGrad, tau, w, h, nc);
    }

    timer2.end(); float t2 = timer2.get();  // elapsed time in seconds

    // copy device memory to host
    cudaMemcpy( imgOut, d_imgIn, nc*w*h*sizeof(float), cudaMemcpyDeviceToHost );

    // free GPU arrays
    cudaFree(d_imgIn);
    cudaFree(d_imgTimeGrad);
    cudaFree(d_imgGrad_x);
    cudaFree(d_imgGrad_y);
    cudaFree(d_imgOut);

    // ###
    // ###
    timer.end();  float t = timer.get();  // elapsed time in seconds

    cout << "kernel time: " << t2*1000 << " ms" << endl;
    cout << "GPU time: " << t*1000 << " ms" << endl;




    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);


    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)


    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
