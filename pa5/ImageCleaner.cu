#include "ImageCleaner.h"

#ifndef SIZEX
#error Please define SIZEX.
#endif
#ifndef SIZEY
#error Please define SIZEY.
#endif

//----------------------------------------------------------------
// TODO:  CREATE NEW KERNELS HERE.  YOU CAN PLACE YOUR CALLS TO
//        THEM IN THE INDICATED SECTION INSIDE THE 'filterImage'
//        FUNCTION.
//
// BEGIN ADD KERNEL DEFINITIONS
//----------------------------------------------------------------

#define debug 0
#define PI	3.14159256
#define DIRECTION_X 1
#define DIRECTION_Y 2
#define FILTER_WIDTH 16


__global__ void fftKernel(float *real_image, float *imag_image, int direction)
{
    // thread block represents a single x
    // each thread in block is computing a particular y

    int row_idx = threadIdx.x;
    int image_index;
    if (direction == DIRECTION_X) {
        image_index = blockIdx.x*SIZEY + threadIdx.x;
    } else {
        image_index = threadIdx.x*SIZEY + blockIdx.x;
    }
    // allocate row y
    __shared__ float real_row[SIZEY];
    __shared__ float imag_row[SIZEY];

    real_row[row_idx] = real_image[image_index];
    imag_row[row_idx] = imag_image[image_index];

    __syncthreads();

    // Compute the value for this index
    float real_value = 0;
    float imag_value = 0;
    for(unsigned int n = 0; n < SIZEY; n++)
    {
        float term = -2 * PI * row_idx * n / SIZEY;
        float real_term = cos(term);
        float imag_term = sin(term);
        real_value += (real_row[n] * real_term) - (imag_row[n] * imag_term);
        imag_value += (imag_row[n] * real_term) + (real_row[n] * imag_term);
    }

    real_image[image_index] = real_value;
    imag_image[image_index] = imag_value;
}

__global__ void ifftKernel(float *real_image, float *imag_image, int direction)
{
    // thread block represents a single x
    // each thread in block is computing a particular y

    int row_idx = threadIdx.x;
    int image_index;
    if (direction == DIRECTION_X) {
        image_index = blockIdx.x*SIZEY + threadIdx.x;
    } else {
        image_index = threadIdx.x*SIZEY + blockIdx.x;
    }
    // allocate row y
    __shared__ float real_row[SIZEY];
    __shared__ float imag_row[SIZEY];

    real_row[row_idx] = real_image[image_index];
    imag_row[row_idx] = imag_image[image_index];

    __syncthreads();

    // Compute the value for this index
    float real_value = 0;
    float imag_value = 0;
    for(unsigned int n = 0; n < SIZEY; n++)
    {
        float term = 2 * PI * row_idx * n / SIZEY;
        float real_term = cos(term);
        float imag_term = sin(term);
        real_value += (real_row[n] * real_term) - (imag_row[n] * imag_term);
        imag_value += (imag_row[n] * real_term) + (real_row[n] * imag_term);
    }

    real_image[image_index] = real_value / SIZEY;
    imag_image[image_index] = imag_value / SIZEY;
}

__global__ void filterKernel(float *real_image, float *imag_image)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int eightX = SIZEX/8;
    int eight7X = SIZEX - eightX;
    int eightY = SIZEY/8;
    int eight7Y = SIZEY - eightY;
    if(!(x < eightX && y < eightY) &&
       !(x < eightX && y >= eight7Y) &&
       !(x >= eight7X && y < eightY) &&
       !(x >= eight7X && y >= eight7Y))
    {
        // Zero out these values
        real_image[y*SIZEX + x] = 0;
        imag_image[y*SIZEX + x] = 0;
    }
}

//----------------------------------------------------------------
// END ADD KERNEL DEFINTIONS
//----------------------------------------------------------------

__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
{
    // check that the sizes match up
    assert(size_x == SIZEX);
    assert(size_y == SIZEY);

    int matSize = size_x * size_y * sizeof(float);

    // These variables are for timing purposes
    float transferDown = 0, transferUp = 0, execution = 0;
    float fftxTime = 0, fftyTime = 0, filterTime = 0, ifftxTime = 0, ifftyTime = 0;
    cudaEvent_t start,stop;

    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    // Create a stream and initialize it
    cudaStream_t filterStream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

    // Alloc space on the device
    float *device_real, *device_imag;
    CUDA_ERROR_CHECK(cudaMalloc((void**)&device_real, matSize));
    CUDA_ERROR_CHECK(cudaMalloc((void**)&device_imag, matSize));

    // Start timing for transfer down
    CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

    // Here is where we copy matrices down to the device 
    CUDA_ERROR_CHECK(cudaMemcpy(device_real,real_image,matSize,cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(device_imag,imag_image,matSize,cudaMemcpyHostToDevice));

    // Stop timing for transfer down
    CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

    // Start timing for the execution
    if (!debug) CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

    //----------------------------------------------------------------
    // TODO: YOU SHOULD PLACE ALL YOUR KERNEL EXECUTIONS
    //        HERE BETWEEN THE CALLS FOR STARTING AND
    //        FINISHING TIMING FOR THE EXECUTION PHASE
    // BEGIN ADD KERNEL CALLS
    //----------------------------------------------------------------

    // This is an example kernel call, you should feel free to create
    // as many kernel calls as you feel are needed for your program
    // Each of the parameters are as follows:
    //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
    //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)
    //    3. Always should be '0' unless you read the CUDA manual and learn about dynamically allocating shared memory
    //    4. Stream to execute kernel on, should always be 'filterStream'
    //
    // Also note that you pass the pointers to the device memory to the kernel call
    if (debug) {

        CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
        fftKernel<<<SIZEY,SIZEX,0,filterStream>>>(device_real, device_imag, DIRECTION_Y);
        CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&fftyTime,start,stop));

        CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
        fftKernel<<<SIZEX,SIZEY,0,filterStream>>>(device_real, device_imag, DIRECTION_X);
        CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&fftxTime,start,stop));

        CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
        filterKernel<<<dim3(SIZEX/FILTER_WIDTH, SIZEY/FILTER_WIDTH),
            dim3(FILTER_WIDTH, FILTER_WIDTH),
            0,
            filterStream>>>
                (device_real, device_imag);
        CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&filterTime,start,stop));

        CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
        ifftKernel<<<SIZEY,SIZEX,0,filterStream>>>(device_real, device_imag, DIRECTION_Y);
        CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&ifftyTime,start,stop));

        CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
        ifftKernel<<<SIZEX,SIZEY,0,filterStream>>>(device_real, device_imag, DIRECTION_X);
        CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&ifftxTime,start,stop));
    } else {
        fftKernel<<<SIZEX,SIZEY,0,filterStream>>>(device_real, device_imag, DIRECTION_X);
        fftKernel<<<SIZEY,SIZEX,0,filterStream>>>(device_real, device_imag, DIRECTION_Y);

        filterKernel<<<dim3(SIZEX/FILTER_WIDTH, SIZEY/FILTER_WIDTH),
            dim3(FILTER_WIDTH, FILTER_WIDTH),
            0,
            filterStream>>>
                (device_real, device_imag);

        ifftKernel<<<SIZEX,SIZEY,0,filterStream>>>(device_real, device_imag, DIRECTION_X);
        ifftKernel<<<SIZEY,SIZEX,0,filterStream>>>(device_real, device_imag, DIRECTION_Y);
    }
    //---------------------------------------------------------------- 
    // END ADD KERNEL CALLS
    //----------------------------------------------------------------

    // Finish timimg for the execution 
    if (debug) {
        execution = fftxTime + fftyTime + filterTime + ifftxTime + ifftyTime;
    } else {
        CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop));
    }

    // Start timing for the transfer up
    CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

    // Here is where we copy matrices back from the device 
    CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

    // Finish timing for transfer up
    CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));

    // Synchronize the stream
    CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream));
    // Destroy the stream
    CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream));
    // Destroy the events
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));

    // Free the memory
    CUDA_ERROR_CHECK(cudaFree(device_real));
    CUDA_ERROR_CHECK(cudaFree(device_imag));

    // Dump some usage statistics
    printf("CUDA IMPLEMENTATION STATISTICS:\n");
    printf("  Host to Device Transfer Time: %f ms\n", transferDown);
    printf("  Kernel(s) Execution Time: %f ms\n", execution);
    if (debug) {
        printf("    fftx   Execution Time: %f ms\n", fftxTime);
        printf("    ffty   Execution Time: %f ms\n", fftyTime);
        printf("    filter Execution Time: %f ms\n", filterTime);
        printf("    ifftx  Execution Time: %f ms\n", ifftxTime);
        printf("    iffty  Execution Time: %f ms\n", ifftyTime);
    }
    printf("  Device to Host Transfer Time: %f ms\n", transferUp);
    float totalTime = transferDown + execution + transferUp;
    printf("  Total CUDA Execution Time: %f ms\n\n", totalTime);
    // Return the total time to transfer and execute
    return totalTime;
}

