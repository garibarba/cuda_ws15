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

// uncomment to use the camera
#define CAMERA

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
    cout << "time: " << t*1000 << " ms" << endl;
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
    cudaMemset( d_imgOut, 0, w*h*nc*sizeof(float) ); CUDA_CHECK;
    // copy image and kernel to device
    cudaMemcpy( d_kernel, kernel, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;
    cudaMemcpy( d_imgIn, imgIn, w*h*nc*sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;
    // launch kernel
    dim3 block = dim3(32,8,1);
    dim3 grid = dim3( (w + block.x -1)/block.x, (h + block.y -1)/block.y, 1);
    gpu_convolution <<<grid,block>>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, kernel_size); CUDA_CHECK;

    // copy to host
    cudaMemcpy( imgOutGPU, d_imgOut, w*h*nc*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;

    // ###
    // ###
    timerg.end();  float tg = timerg.get();  // elapsed time in seconds
#ifndef CAMERA
    cout << "time: " << tg*1000 << " ms" << endl;
#endif






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
