PA5 (1 late day)
John Doherty (doherty1)

For this assignment I used three kernels:
- fftKernel: This kernel creates a thread block for each pixel along the dimension of the image over which we are performing the transform. That thread block is then composed of threads that all compute a single pixel value along that dimension. This works because each thread in the thread block will be operating on the same row or column in the image. Thus we can put that row or column into an array in the block's shared memory. Each thread will compute the value of one pixel in the transformed image by iterating over the values in the shared array. Since every thread is writing to a single pixel in the new image in global memory, there are no conflicts.

- ifftKernel: This kernel operates in the same way as the fftKernel, but performs a slightly different computation to compute the inverse transform.

- filterKernel: This kernel performs the low-pass filtering. I break the image into 16x16 blocks and give each block to a thread block. There are 16x16 threads in each thread block and each computes the value of a pixel.
