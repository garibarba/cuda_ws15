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

// imgIn CUDA texture has to be declared before any used
texture <float,2,cudaReadModeElementType> texRef_imgIn; // at file scope
// constant memory must have fixed size at compile time
// #define KERNEL_SIZE = 1681 // (2*20 + 1)*(2*20 + 1)
__constant__ float constKernel[1681];


__global__ void gpu_convolution(float *imgIn, float *imgOut, float *kernel, int w, int h, int nc, int ks, bool constant_kernel) {
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
        if (x < w && y < h) {
          if (constant_kernel) {
            imgOut[x + w*y + w*h*c] += imgIn[img_x + w*img_y + w*h*c] * constKernel[k + ks*l];
          } else {
            imgOut[x + w*y + w*h*c] += imgIn[img_x + w*img_y + w*h*c] * kernel[k + ks*l];
          }
        }
      }
    }
  }
}

__global__ void gpu_convolution_sharedmem(float *imgIn, float *imgOut, float *kernel, int w, int h, int nc, int ks, bool constant_kernel) { // with imgIn and kernel in global
  // image indexes under a mask
  int img_x, img_y;

  // calculate main pixel corresponding to thread
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  // calculate thread indexes in block coordinates
  int xblock = threadIdx.x;
  int yblock = threadIdx.y;
  int lin_threadIdx = xblock + blockDim.x * yblock; // index in a linearized array of indexes
  // allocate shared array
  extern __shared__ float sh_img[];
  //load array values
    // calculate size of sh_img again ?????????????????????
  int shblock_w = blockDim.x + ks - 1;
  int shblock_h = blockDim.y + ks - 1;
  int shblock_size = shblock_w * shblock_h;
  int shblock_topleft_x = blockDim.x * blockIdx.x - ks/2; // sometimes negative
  int shblock_topleft_y = blockDim.y * blockIdx.y - ks/2;
    // number of threads in the block
  int num_threads = blockDim.x * blockDim.y;
  int num_loads = (shblock_size + num_threads - 1) / num_threads;
    // shared block coordinates
  int x_sh, y_sh;
  int idx_sh;
    //for every channel
  for (int c = 0; c < nc; c++) {
    // each thread loads some data
    for (int l = 0; l < num_loads; l++) {
      idx_sh = lin_threadIdx + l*num_threads;
      if (idx_sh < shblock_size) {
        // if (c == 0 && blockIdx.x == 0 && blockIdx.y == 0) printf("%d %d %d \n", num_threads, shblock_size, idx_sh);
        // if (c == 0 && blockIdx.x == 0 && blockIdx.y == 0) printf("xblock: %d yblock: %d blockDim.y: %d thread: %d \n", xblock, yblock, blockDim.y, lin_threadIdx);
        img_x = min(w-1, max(0, shblock_topleft_x + idx_sh % shblock_w));
        img_y = min(h-1, max(0, shblock_topleft_y + idx_sh / shblock_w));
        sh_img[idx_sh] = imgIn[img_x + img_y*w + c*w*h]; // imgIn in global
      }
    }

    // wait for all to finish copying
    __syncthreads();

    // for every pixel in the kernel
    for (int k = 0; k < ks; ++k) {
      for (int l = 0; l < ks; ++l) {
        x_sh = xblock + k;
        y_sh = yblock + l;
        if (x < w && y < h) {
          if (constant_kernel) {
            imgOut[x + w*y + w*h*c] += sh_img[x_sh + y_sh*shblock_w] * constKernel[k + ks*l];
          } else {
            imgOut[x + w*y + w*h*c] += sh_img[x_sh + y_sh*shblock_w] * kernel[k + ks*l];
          }
        }
      }
    }
    // wait for the channel ops to finish
    __syncthreads();
  }
}

__global__ void gpu_convolution_texturemem(float *imgOut, float *kernel, int w, int h, int nc, int ks,  bool constant_kernel) {
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
        if (x < w && y < h) {
          if (constant_kernel)
            imgOut[x + w*y + w*h*c] += tex2D(texRef_imgIn, img_x + 0.5f, img_y + c*h + 0.5f) * constKernel[k + ks*l]; // imgIn in texture. 0.5 to get the center of the pixel
          else
            imgOut[x + w*y + w*h*c] += tex2D(texRef_imgIn, img_x + 0.5f, img_y + c*h + 0.5f) * kernel[k + ks*l]; // imgIn in texture. 0.5 to get the center of the pixel
        }
      }
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
    float sigma = 15;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << " with kernel size of 2*ceil(3*sigma) + 1" << endl;
    int kernel_size = 2*ceil(3*sigma) + 1; // directly defined by sigma

    // load the kernel into constant memory if "-constant_kernel" is specified
    bool constant_kernel = false;
    getParam("constant_kernel", constant_kernel, argc, argv);
    cout << "constant_kernel: " << constant_kernel << endl;
    if (constant_kernel) cout << "warning! constant_kernel only has enough memory for 3*sigma <= 20" << endl;

    // load the input image into texture memory if "-texture_imgin" is specified
    bool texture_imgin = false;
    getParam("texture_imgin", texture_imgin, argc, argv);
    cout << "texture_imgin: " << texture_imgin << endl;

    // load the input image into texture memory if "-texture_imgin" is specified
    bool shared_imgin = false;
    getParam("shared_imgin", shared_imgin, argc, argv);
    cout << "shared_imgin: " << shared_imgin << endl;

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
    cv::Mat mKer(kernel_size, kernel_size, CV_32FC1);
    // GPU Output image
    cv::Mat mOutGPU(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers


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
    float *imgOutGPU = new float[(size_t)w*h*mOut.channels()];



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



    // define kernel through CPU function
    float *kernel = gaussian_kernel(kernel_size, sigma);

  #ifndef CAMERA
    // CPU time
    Timer timer; timer.start();
    // ###
    // ###
    convolution(imgIn, imgOut, kernel, w, h, nc, kernel_size);
    // cout << "-----------" << endl;
    // for (int i = 0; i < kernel_size; i++) {
    //   for (int j = 0; j < kernel_size; j++) {
    //     cout << kernel[i + kernel_size *j] << endl;
    //   }
    // }
    // ###
    // ###
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "CPU time: " << t*1000 << " ms" << endl;
  #endif


    // GPU time
    Timer timerg; timerg.start();
    // ###
    // ###
    // initialize device memory
    float *d_kernel = NULL;
    float *d_imgIn = NULL;
    float *d_imgOut = NULL;
    cudaMalloc( &d_kernel, kernel_size*kernel_size*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgIn, w*h*nc*sizeof(float) ); CUDA_CHECK;
    cudaMalloc( &d_imgOut, w*h*nc*sizeof(float) ); CUDA_CHECK;
    // fill device imgOut with 0s to be able to add values directly to it
    cudaMemset( d_imgOut, 0, w*h*nc*sizeof(float) ); CUDA_CHECK;

    // copy imgIn to device global memory
    cudaMemcpy( d_imgIn, imgIn, w*h*nc*sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;
    // bind imgIn in global memory to texture memory
    if (texture_imgin) {
      texRef_imgIn.addressMode[0] = cudaAddressModeClamp;
      texRef_imgIn.addressMode[1] = cudaAddressModeClamp;
      texRef_imgIn.filterMode = cudaFilterModeLinear; // linear interpolation
      texRef_imgIn.normalized = false; // Set whether coordinates are normalized to [0,1)
      cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); // number of bits for each texture channel
      cudaBindTexture2D(NULL, &texRef_imgIn, d_imgIn, &desc, w, h*nc, w*sizeof(d_imgIn[0]));
    }
    // put kernel in some device memory
    if (constant_kernel) {
      cudaMemcpyToSymbol (constKernel, kernel, kernel_size*kernel_size*sizeof(float)); CUDA_CHECK; // kernel in constant
    } else {
      cout << "here!" << endl;
      cudaMemcpy( d_kernel, kernel, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK; // kernel in global
    }
    // launch kernel
    dim3 block = dim3(32,8,1);
    cout << "Blocksize: " << block.x << "x" << block.y << endl;
    dim3 grid = dim3( (w + block.x -1)/block.x, (h + block.y -1)/block.y, 1);
    size_t smBytes = (block.x + kernel_size - 1) * (block.y + kernel_size - 1) * sizeof(float); // only per channel. Take advantage through loop
    cout << "Shared memory bytes: " << smBytes << endl;
    // WARNING
    if (texture_imgin && shared_imgin)
      cout << "!!! Enabling both texture and shared options results in running with texture" << endl;
    if (texture_imgin) {
      gpu_convolution_texturemem <<<grid,block>>> (d_imgOut, d_kernel, w, h, nc, kernel_size, constant_kernel); CUDA_CHECK; // with imgIn and kernel in global
    } else if (shared_imgin) { // shared memory
      gpu_convolution_sharedmem <<<grid,block,smBytes>>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, kernel_size, constant_kernel); CUDA_CHECK; // with imgIn and kernel in global
    } else {
      gpu_convolution <<<grid,block,smBytes>>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, kernel_size, constant_kernel); CUDA_CHECK; // with imgIn and kernel in global
    }

    // copy to host
    cudaMemcpy( imgOutGPU, d_imgOut, w*h*nc*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;

    // ###
    // ###
    timerg.end();  float tg = timerg.get();  // elapsed time in seconds
#ifndef CAMERA
    cout << "GPU time: " << tg*1000 << " ms" << endl;
#endif


    // free memory
    cudaFree( d_kernel );
    cudaFree( d_imgIn );
    cudaFree( d_imgOut );



    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

#ifndef CAMERA
    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("CPU Output", mOut, 100+5+kernel_size+5+w+40, 100);
#endif

    // show GPU output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOutGPU, imgOutGPU);
    showImage("GPU Output", mOutGPU, 100+5+kernel_size+5+w+40+w+40, 100);

    // ### Display your own output images here as needed

    // show kernel image
    convert_layered_to_mat(mKer, kernel);
    // double min, max;
    // cv::minMaxLoc(mKer, &min, &max);
    showImage("Kernel", mKer/kernel[kernel_size*kernel_size/2], 100+w+5, 100); // mKer is upscaled with its largest value for visualization


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

// clamp an index to the min and max values specified
int clamp(int idx, int minval, int maxval) {
  // int clamped_idx = idx;
  // if (idx < min) clamped_idx = min;
  // else if (idx > max) clamped_idx = max;
  // return clamped_idx;
  return min(maxval, max(idx, minval));
}
