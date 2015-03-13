#ifndef __IMAGE_CLEANER__
#define __IMAGE_CLEANER__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"


// A macro for checking the error codes of cuda runtime calls
#define CUDA_ERROR_CHECK(expr) \
  {                            \
    cudaError_t err = expr;    \
    if (err != cudaSuccess)    \
    {                          \
      printf("CUDA call failed!\n%s\n", cudaGetErrorString(err)); \
      exit(1);                 \
    }                          \
  }



// This is the entry point function that has to be filled out for this assignment
// The float value that it returns is the total time taken to transfer the data
// to the device, execute all kernels, and transfer the data back from the device
float filterImage(float *real_image, float *imag_image, int size_x, int size_y);

#endif
