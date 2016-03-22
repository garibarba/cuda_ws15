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
// #define CAMERA

// clamp an index to the min and max values specified
int clamp(int idx, int min, int max);

float* gaussian_kernel(int kernel_size, float sigma) {
  float *kernel = new float[kernel_size * kernel_size];

  float mid = (float)kernel_size/2.f; // coordinate value of the center ok the kernel
  float dist_sq;
  float norm_sum = 0; // normalization factor
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      dist_sq = powf((float)i + 0.5 - mid, 2) + powf((float)j + 0.5 - mid, 2);
      kernel[i + kernel_size * j] = expf( - dist_sq / (2*powf(sigma, 2)) );
      norm_sum += kernel[i + kernel_size * j];
    }
  }
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      kernel[i + kernel_size * j] /= norm_sum;
      // cout << kernel[i + kernel_size *j] << endl;
    }
  }
  return kernel;
}

void convolution(float *imgIn, float *imgOut, float *kernel, int w, int h, int nc, int ks) {
  int img_x, img_y;
  // for every channel
  for (int c = 0; c < nc; c++) {
    // for every pixel in the image
    for (int i = 0; i < w; i++){
      for (int j = 0; j < h; j++) {
        // for every pixel in the kernel
        for (int k = 0; k < ks; k++) {
          for (int l = 0; l < ks; l++) {
            img_x = clamp(i + k - (ks/2 + 1), 0, w-1);
            img_y = clamp(j + l - (ks/2 + 1), 0, h-1);
            imgOut[i + w*j + w*h*c] += imgIn[img_x + w*img_y + w*h*c] * kernel[k + ks*l];
          }
        }
      }
    }
  }
}

__global__ void gpu_convolution(float *imgIn, float *imgOut, float *kernel, int w, int h, int nc, int ks) {
  // indexes for the kernel
  int img_x, img_y;
  // calculate center pixel corresponding to thread
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  // for every channel
  for (int c = 0; c < nc; ++c) {
    // for every pixel in the kernel
    for (int k = 0; k < ks; ++k) {
      for (int l = 0; l < ks; ++l) {
        img_x = min(w-1, max(0, x + k - (ks/2 + 1)));
        img_y = min(h-1, max(0, y + l - (ks/2 + 1)));
        if (x < w && y < h) imgOut[x + w*y + w*h*c] += imgIn[img_x + w*img_y + w*h*c] * kernel[k + ks*l];
      }
    }
  }
}

__device__ int d_clamp(int ind, int minval, int maxval) {
  return min(max(minval, ind), maxval);
}

__global__ void gradient(float *d_imgIn, float *d_imgGrad_x, float *d_imgGrad_y, int w, int h, int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int ind = x + w * y;
  if ( x < w && y < h ) {
    for (int c = 0; c < nc; c++) {
      if (x + 1 < w) d_imgGrad_x[ind + c*w*h] = d_imgIn[ind + c*w*h + 1] - d_imgIn[ind + c*w*h]; // derivative along x
      else           d_imgGrad_x[ind + c*w*h] = 0;
      if (y + 1 < h) d_imgGrad_y[ind + c*w*h] = d_imgIn[ind + c*w*h + w] - d_imgIn[ind + c*w*h]; // derivative along y
      else           d_imgGrad_y[ind + c*w*h] = 0;
    }
  }
}

__global__ void ri_gradient(float *d_imgIn, float *d_imgGrad_x, float *d_imgGrad_y, int w, int h, int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  for (int c = 0; c < nc; c++) {
    if (x < w && y < h) {
      d_imgGrad_x[x + y*w + c*w*h] =
                      ( 3*d_imgIn[d_clamp(x+1,0,w-1) + w*d_clamp(y-1,0,h-1) + c*w*h]
                        + 10*d_imgIn[d_clamp(x+1,0,w-1) + w*d_clamp(y,0,h-1) + c*w*h]
                        + 3*d_imgIn[d_clamp(x+1,0,w-1) + w*d_clamp(y+1,0,h-1) + c*w*h]
                        - 3*d_imgIn[d_clamp(x-1,0,w-1) + w*d_clamp(y-1,0,h-1) + c*w*h]
                        - 10*d_imgIn[d_clamp(x-1,0,w-1) + w*d_clamp(y,0,h-1) + c*w*h]
                        - 3*d_imgIn[d_clamp(x-1,0,w-1) + w*d_clamp(y+1,0,h-1) + c*w*h] ) / 32.f; // derivative along x
      d_imgGrad_y[x + y*w + c*w*h] =
                      ( 3*d_imgIn[d_clamp(x-1,0,w-1) + w*d_clamp(y+1,0,h-1) + c*w*h]
                        + 10*d_imgIn[d_clamp(x,0,w-1) + w*d_clamp(y+1,0,h-1) + c*w*h]
                        + 3*d_imgIn[d_clamp(x+1,0,w-1) + w*d_clamp(y+1,0,h-1) + c*w*h]
                        - 3*d_imgIn[d_clamp(x-1,0,w-1) + w*d_clamp(y-1,0,h-1) + c*w*h]
                        - 10*d_imgIn[d_clamp(x,0,w-1) + w*d_clamp(y-1,0,h-1) + c*w*h]
                        - 3*d_imgIn[d_clamp(x+1,0,w-1) + w*d_clamp(y-1,0,h-1) + c*w*h] ) / 32.f; // derivative along x
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
  if ( x < w && y < h ) {
    for (int c = 0; c < nc; c++) {
      if (x > 0) dx = d_imgV1[ind + (size_t)c*w*h] - d_imgV1[ind + (size_t)c*w*h - 1]; // derivative along x
      else       dx = d_imgV1[ind + (size_t)c*w*h];
      if (y > 0) dy = d_imgV2[ind + (size_t)c*w*h] - d_imgV2[ind + (size_t)c*w*h - w]; // derivative along y
      else       dy = d_imgV2[ind + (size_t)c*w*h];
      d_div[ind + (size_t)c*w*h] = dx + dy;
    }
  }
}

__global__ void gpu_m_product(float *imgV1, float *imgV2, float *imgM11, float *imgM12, float *imgM22, int w, int h, int nc) {
  // calculate center pixel corresponding to thread
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  // for every channel
  for (int c = 0; c < nc; c++) {
    if (x < w && y < h) {
      imgM11[x + y*w] += imgV1[x + y*w + c*w*h]*imgV1[x + y*w + c*w*h];
      imgM12[x + y*w] += imgV1[x + y*w + c*w*h]*imgV2[x + y*w + c*w*h];
      imgM22[x + y*w] += imgV2[x + y*w + c*w*h]*imgV2[x + y*w + c*w*h];
    }
  }
}

__device__ void eigval(float *eigenval, float m11, float m12, float m21, float m22) {
  // implementing solution for: lambda**2 + lambda*(-m22-m11) + (a11*a22)
  // a = 1
  float b = - m11 - m22;
  float c = m11*m22 - m12*m21;
  eigenval[0] = (-b - sqrtf(b*b - 4*c))/2;
  eigenval[1] = (-b + sqrtf(b*b - 4*c))/2;
}

// calculate the norm of a 2D vector
__host__ __device__ float norm_2d(float v1, float v2) {
  return sqrtf(v1*v1 + v2*v2);
}

// only valid for 2d matrices
__device__ void eig(float *eigenval, float *eigenvec, float m11, float m12, float m21, float m22) {
  // eigenvalues
  // implementing solution for: lambda**2 + lambda*(-m22-m11) + (a11*a22)
  // a*lambda**2 + b*lambda + c = 0
  float b = - m11 - m22;
  float c = m11*m22 - m12*m21;
  eigenval[0] = (-b - sqrtf(b*b - 4.f*c))/2.f;
  eigenval[1] = (-b + sqrtf(b*b - 4.f*c))/2.f;

  // eigenvectors
  float a; float d;
  for (int i = 0; i < 2; i++) {
    // now abcd are the elements of the matrix (A-lambda*I)
    a = m11 - eigenval[i];
    b = m12;
    c = m21;
    d = m22 - eigenval[i];

    if ( a*a > 0 ) {
      if ( b*b > 0 ) {
        eigenvec[0 + 2*i] = 1 / norm_2d(1.f, a/b);
        eigenvec[1 + 2*i] = -(a/b) / norm_2d(1.f, a/b);
      }
    } else if ( c*c > 0 ) {
      if ( d*d > 0 ) {
        eigenvec[0 + 2*i] = 1 / norm_2d(1.f, c/d);
        eigenvec[1 + 2*i] = -(c/d) / norm_2d(1.f, c/d);
      }
    } else {
      printf("!!! eig ");
      eigenvec[0 + 2*i] = 1;
      eigenvec[1 + 2*i] = 1;
    }
  }
}

// only 2x2 matrices
// returns the G tensor for anisotropic Difussion
__device__ void G_tensor(float *G, float m11, float m12, float m21, float m22, float alpha, float C) {
  // get eigenvalues
  float eigenval[2];
  float eigenvec[4];
  eig(eigenval, eigenvec, m11, m12, m21, m22);
  // mu factors
  float mu[2];
  mu[0] = alpha;
  if ( (eigenval[0] - eigenval[1])*(eigenval[0] - eigenval[1]) < 1e-10 ) {
    printf("!!! muuu ");
    mu[1] = alpha;
  }
  else {
    mu[1] = alpha + (1 - alpha)*exp( - C /( (eigenval[0] - eigenval[1])*(eigenval[0] - eigenval[1]) ) );
    if (blockIdx.x == 0 && blockIdx.y ==0) printf("%4.2f", mu[1]);
  }
  // calculate G
  // this is originating a nan
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      G[i + 2*j] = mu[0]*eigenvec[i]*eigenvec[j] + mu[1]*eigenvec[i + 2]*eigenvec[j + 2];
    }
  }
}

__global__ void calc_G_tensor(float *imgG, float *imgT11, float *imgT12, float *imgT22, int w, int h, int nc, float alpha, float C) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x < w && y < h) {
    int idx;
    float G[4];// = {0};
    // for every channel
    for (int c = 0; c < nc; c++) {
      idx = x + y*w;
      G_tensor(G, imgT11[idx], imgT12[idx], imgT12[idx], imgT22[idx], alpha, C);
      // for each of the 4 tensor components
      for (int i = 0; i < 4; i++) {
        // if ( c==0 && x%10==0 && y%10==0 && i==0) printf("%4.2f \n", G[0]);
        imgG[idx + c*w*h + i*nc*w*h] = G[i];
      }
    }
  }
}

__global__ void tensor_scaling(float *imgV1, float *imgV2, float *imgG, int w, int h, int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x < w && y < h) {
    int idx;
    float v1, v2;
    // for every channel
    for (int c = 0; c < nc; c++) {
      idx = x + y*w;
      v1 = imgV1[idx + c*w*h];
      v2 = imgV2[idx + c*w*h];
      imgV1[idx + c*w*h] = imgG[idx + c*w*h + 0*nc*w*h]*v1 + imgG[idx + c*w*h + 1*nc*w*h]*v2;
      imgV2[idx + c*w*h] = imgG[idx + c*w*h + 2*nc*w*h]*v1 + imgG[idx + c*w*h + 3*nc*w*h]*v2;
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

    // load the value for sigma if "-sigma" is specified
    float sigma = 0.5;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << " with smoothing kernel size of 2*ceil(3*sigma) + 1" << endl;
    int kernel_size_sigma = 2*ceil(3*sigma) + 1; // directly defined by sigma

    // load the value for ro if "-ro" is specified
    float ro = 3;
    getParam("ro", ro, argc, argv);
    cout << "ro: " << ro << " with averaging kernel size of 2*ceil(3*ro) + 1" << endl;
    int kernel_size_ro = 2*ceil(3*sigma) + 1; // directly defined by sigma

    // G diffusion tensor parameter
    float alpha = 0.01;
    getParam("alpha", alpha, argc, argv);
    cout << "alpha: " << alpha << endl;

    // G diffusion tensor parameter
    float C = 0.000005;
    getParam("C", C, argc, argv);
    cout << "C: " << C << endl;

    // number of time steps
    int N = 10;
    getParam("N", N, argc, argv);
    cout << "N: " << N << endl;

    // size of time steps
    float tau = 0.25;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    cout << "tau x N = " << tau*N << ", if anisotropic is equivalent to sigma = " << sqrt(2*tau*N) << endl;


    cout << "--------------" << endl; // save our eyes

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
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed

    // Set the OpenCV kernel display image
    cv::Mat mKer(kernel_size_sigma, kernel_size_sigma, CV_32FC1);
    // structure tensor grayscale Output image
    cv::Mat mOutMii(h,w,CV_32FC1);  // mOutMii will have just one channel
    // debugging aux
    cv::Mat mAux(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers


    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    // allocate raw output array for the GPU
    float *imgOutTii = new float[(size_t)w*h*mOutMii.channels()];

    // auxiliar for debugging
    float *imgAux = new float[(size_t)w*h*mOut.channels()];



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




    float *kernel_sigma = gaussian_kernel(kernel_size_sigma, sigma);
    float *kernel_ro = gaussian_kernel(kernel_size_ro, ro);

  // #ifndef CAMERA
  //   // CPU time
  //   Timer timer; timer.start();
  //   // ###
  //   // ###
  //   convolution(imgIn, imgOut, kernel, w, h, nc, kernel_size);
  //   // cout << "-----------" << endl;
  //   // for (int i = 0; i < kernel_size; i++) {
  //   //   for (int j = 0; j < kernel_size; j++) {
  //   //     cout << kernel[i + kernel_size *j] << endl;
  //   //   }
  //   // }
  //   // ###
  //   // ###
  //   timer.end();  float t = timer.get();  // elapsed time in seconds
  //   cout << "time: " << t*1000 << " ms" << endl;
  // #endif


    // GPU time
    Timer timerg; timerg.start();
    // ###
    // ###
    // initialize device memory
    float *d_kernel_sigma = NULL;
    float *d_kernel_ro = NULL;
    float *d_imgIn = NULL;
    float *d_imgG = NULL;
    float *d_imgTimeGrad = NULL;
    float *d_imgV1 = NULL;
    float *d_imgV2 = NULL;
    float *d_imgS = NULL;
    float *d_imgM11 = NULL;
    float *d_imgM12 = NULL;
    float *d_imgM22 = NULL;
    float *d_imgT11 = NULL;
    float *d_imgT12 = NULL;
    float *d_imgT22 = NULL;
    float *d_imgOut = NULL;
    cudaMalloc( &d_kernel_sigma, kernel_size_sigma*kernel_size_sigma*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_kernel_ro, kernel_size_ro*kernel_size_ro*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgIn, w*h*nc*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgG, 4*w*h*nc*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgTimeGrad, w*h*nc*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgV1, w*h*nc*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgV2, w*h*nc*sizeof(float) ); CUDA_CHECK;

    cudaMalloc( &d_imgS, w*h*nc*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgM11, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgM12, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgM22, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgT11, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgT12, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgT22, w*h*sizeof(float) ); CUDA_CHECK;

    cudaMalloc( &d_imgOut, w*h*nc*sizeof(float) ); CUDA_CHECK;


    cudaMemset( d_imgS, 0, w*h*nc*sizeof(float) ); CUDA_CHECK;
    cudaMemset( d_imgM11, 0, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMemset( d_imgM12, 0, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMemset( d_imgM22, 0, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMemset( d_imgT11, 0, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMemset( d_imgT12, 0, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMemset( d_imgT22, 0, w*h*sizeof(float) ); CUDA_CHECK;
    cudaMemset( d_imgOut, 0, w*h*nc*sizeof(float) ); CUDA_CHECK;
    // copy image and kernel to device
    cudaMemcpy( d_kernel_sigma, kernel_sigma, kernel_size_sigma*kernel_size_sigma*sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;
    cudaMemcpy( d_kernel_ro, kernel_ro, kernel_size_ro*kernel_size_ro*sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;
    cudaMemcpy( d_imgIn, imgIn, w*h*nc*sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;
    // launch kernel
    dim3 block = dim3(32,8,1);
    dim3 grid = dim3( (w + block.x -1)/block.x, (h + block.y -1)/block.y, 1);

    // G tensor is only calculated once
    gpu_convolution <<<grid,block>>> (d_imgIn, d_imgS, d_kernel_sigma, w, h, nc, kernel_size_ro); CUDA_CHECK;
    gradient <<<grid,block>>> (d_imgS, d_imgV1, d_imgV2, w, h, nc); CUDA_CHECK;
    gpu_m_product <<<grid,block>>> (d_imgV1, d_imgV2, d_imgM11, d_imgM12, d_imgM22, w, h, nc); CUDA_CHECK;
    gpu_convolution <<<grid,block>>> (d_imgM11, d_imgT11, d_kernel_ro, w, h, 1, kernel_size_ro); CUDA_CHECK;
    gpu_convolution <<<grid,block>>> (d_imgM12, d_imgT12, d_kernel_ro, w, h, 1, kernel_size_ro); CUDA_CHECK;
    gpu_convolution <<<grid,block>>> (d_imgM22, d_imgT22, d_kernel_ro, w, h, 1, kernel_size_ro); CUDA_CHECK;
    calc_G_tensor <<<grid,block>>> (d_imgG, d_imgT11, d_imgT12, d_imgT22, w, h, nc, alpha, C);
    // for a lot of time steps
    for (int n = 0; n < N; n++) {
      gradient <<<grid,block>>> (d_imgS, d_imgV1, d_imgV2, w, h, nc); CUDA_CHECK;
      tensor_scaling <<<grid,block>>> (d_imgV1, d_imgV2, d_imgG, w, h, nc); CUDA_CHECK;
      divergence_2d <<<grid,block>>> (d_imgV1, d_imgV2, d_imgTimeGrad, w, h, nc); CUDA_CHECK;
      time_step <<<grid,block>>> (d_imgS, d_imgTimeGrad, tau, w, h, nc);
    }



    // ###
    // ###
    timerg.end();  float tg = timerg.get();  // elapsed time in seconds
#ifndef CAMERA
    cout << "time: " << tg*1000 << " ms" << endl;
#endif

#ifndef CAMERA
    double minval, maxval;
    // // show structure tensor
    // cudaMemcpy( imgOutTii, d_imgT11, w*h*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
    // convert_layered_to_mat(mOutMii, imgOutTii);
    // cv::minMaxLoc(mOutMii, &minval, &maxval);
    // showImage("GPU M11 (Structure Tensor)", mOutMii/maxval, 50 + w, 100);
    //
    // cudaMemcpy( imgOutTii, d_imgT12, w*h*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
    // convert_layered_to_mat(mOutMii, imgOutTii);
    // cv::minMaxLoc(mOutMii, &minval, &maxval);
    // showImage("GPU M12 (Structure Tensor)", mOutMii/maxval, 50 + 2*w, 100);
    //
    // cudaMemcpy( imgOutTii, d_imgT22, w*h*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
    // convert_layered_to_mat(mOutMii, imgOutTii);
    // cv::minMaxLoc(mOutMii, &minval, &maxval);
    // showImage("GPU M22 (Structure Tensor)", mOutMii/maxval, 50 + 3*w, 100);
#endif


    // show input image
    showImage("Input", mIn, 50, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    cudaMemcpy( imgOut, d_imgS, nc*w*h*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
    convert_layered_to_mat(mOut, imgOut);
    showImage("GPU Result", mOut, 50 + w/2, 100 + h);

    // ### Display your own output images here as needed

    // show output image: first convert to interleaved opencv format from the layered raw array
    cudaMemcpy( imgAux, d_imgG, nc*w*h*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
    convert_layered_to_mat(mAux, imgAux);
    cv::minMaxLoc(mAux, &minval, &maxval);
    showImage("GPU Aux image", mAux, 50 + w/2 + w, 100 + h);

    // show kernel image
    convert_layered_to_mat(mKer, kernel_sigma);
    // double min, max;
    // cv::minMaxLoc(mKer, &min, &max);
    showImage("Kernel sigma", mKer/kernel_sigma[kernel_size_sigma*kernel_size_sigma/2], 50 - kernel_size_sigma, 100); // mKer is upscaled with its largest value for visualization

    // free device memory
    cudaFree(d_kernel_sigma);
    cudaFree(d_kernel_ro);
    cudaFree(d_imgIn);
    cudaFree(d_imgTimeGrad);
    cudaFree(d_imgG);
    cudaFree(d_imgV1);
    cudaFree(d_imgV2);
    cudaFree(d_imgS);
    cudaFree(d_imgM11);
    cudaFree(d_imgM12);
    cudaFree(d_imgM22);
    cudaFree(d_imgT11);
    cudaFree(d_imgT12);
    cudaFree(d_imgT22);
    cudaFree(d_imgOut);


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
    delete[] imgOutTii;


    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}

// clamp an index to the min and max values specified
int clamp(int idx, int minval, int maxval) {
  // int clamped_idx = idx;
  // if (idx < min) clamped_idx = min;
  // else if (idx > max) clamped_idx = max;
  // return clamped_idx;
  return min(maxval, max(idx, minval));
}
