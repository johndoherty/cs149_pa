#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>

#define PI	3.14159265

void cpu_fftx(float *real_image, float *imag_image, int size_x, int size_y)
{
    // Local values
    float *fft_real = new float[size_y * size_y];
    float *fft_imag = new float[size_y * size_y];
    unsigned int y = 0;
    unsigned int x = 0;
    unsigned int n = 0;
    #pragma omp parallel for \
            shared(size_x, size_y, fft_real, fft_imag) \
            private(y, n)
    for (y = 0; y < size_y; y++)
    {
        for(n = 0; n < size_y; n++)
        {
            float term = -2 * PI * y * n / size_y;
            fft_real[y*size_y + n] = cos(term);
            fft_imag[y*size_y + n] = sin(term);
        }
    }

    #pragma omp parallel for \
            shared(size_x, size_y, fft_real, fft_imag) \
            private(x, y, n)
    for(x = 0; x < size_x; x++)
    {
        // Create some space for storing temporary values
        float *realOutBuffer = new float[size_y];
        float *imagOutBuffer = new float[size_y];
        for(y = 0; y < size_y; y++)
        {
            // Compute the frequencies for this index
                        // Compute the value for this index
            realOutBuffer[y] = 0.0f;
            imagOutBuffer[y] = 0.0f;
            for(n = 0; n < size_y; n+=2)
            {
                realOutBuffer[y] += (real_image[x*size_y + n] * fft_real[y*size_y + n]) - (imag_image[x*size_y + n] * fft_imag[y*size_y + n]);
                imagOutBuffer[y] += (imag_image[x*size_y + n] * fft_real[y*size_y + n]) + (real_image[x*size_y + n] * fft_imag[y*size_y + n]);
                realOutBuffer[y] += (real_image[x*size_y + n+1] * fft_real[y*size_y + n + 1]) - (imag_image[x*size_y + n + 1] * fft_imag[y*size_y + n + 1]);
                imagOutBuffer[y] += (imag_image[x*size_y + n+1] * fft_real[y*size_y + n + 1]) + (real_image[x*size_y + n + 1] * fft_imag[y*size_y + n + 1]);
            }
        }
        // Write the buffer back to were the original values were
        for(unsigned int y = 0; y < size_y; y+=2)
        {
            real_image[x*size_y + y] = realOutBuffer[y];
            imag_image[x*size_y + y] = imagOutBuffer[y];
            real_image[x*size_y + y + 1] = realOutBuffer[y + 1];
            imag_image[x*size_y + y + 1] = imagOutBuffer[y + 1];
        }
        delete [] realOutBuffer;
        delete [] imagOutBuffer;
    }
    // Reclaim some memory
    delete [] fft_real;
    delete [] fft_imag;
    return;
}

// This is the same as the thing above, except it has a scaling factor added to it
void cpu_ifftx(float *real_image, float *imag_image, int size_x, int size_y)
{
    // Create some space for storing temporary values
    float *realOutBuffer = new float[size_x];
    float *imagOutBuffer = new float[size_x];
    float *fft_real = new float[size_y * size_y];
    float *fft_imag = new float[size_y * size_y];
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int n = 0;
    #pragma omp parallel for \
            shared(size_x, size_y, fft_real, fft_imag) \
            private(y, n)
    for(y = 0; y < size_y; y++)
    {
        for(n = 0; n < size_y; n++)
        {
            // Compute the frequencies for this index
            float term = 2 * PI * y * n / size_y;
            fft_real[y*size_y + n] = cos(term);
            fft_imag[y*size_y + n] = sin(term);
        }
    }

    #pragma omp parallel for \
            shared(size_x, size_y, fft_real, fft_imag) \
            private(x, y, n)
    for(unsigned int x = 0; x < size_x; x++)
    {
        float *realOutBuffer = new float[size_y];
        float *imagOutBuffer = new float[size_y];
        for(unsigned int y = 0; y < size_y; y++)
        {
            // Compute the value for this index
            realOutBuffer[y] = 0.0f;
            imagOutBuffer[y] = 0.0f;
            for(unsigned int n = 0; n < size_y; n+=2)
            {
                realOutBuffer[y] += (real_image[x*size_y + n] * fft_real[y*size_y + n]) - (imag_image[x*size_y + n] * fft_imag[y*size_y + n]);
                imagOutBuffer[y] += (imag_image[x*size_y + n] * fft_real[y*size_y + n]) + (real_image[x*size_y + n] * fft_imag[y*size_y + n]);
                realOutBuffer[y] += (real_image[x*size_y + n + 1] * fft_real[y*size_y + n + 1]) - (imag_image[x*size_y + n + 1] * fft_imag[y*size_y + n + 1]);
                imagOutBuffer[y] += (imag_image[x*size_y + n + 1] * fft_real[y*size_y + n + 1]) + (real_image[x*size_y + n + 1] * fft_imag[y*size_y + n + 1]);
            }

            // Incoporate the scaling factor here
            realOutBuffer[y] /= size_y;
            imagOutBuffer[y] /= size_y;
        }
        // Write the buffer back to were the original values were
        for(unsigned int y = 0; y < size_y; y+=2)
        {
            real_image[x*size_y + y] = realOutBuffer[y];
            imag_image[x*size_y + y] = imagOutBuffer[y];
            real_image[x*size_y + y + 1] = realOutBuffer[y + 1];
            imag_image[x*size_y + y + 1] = imagOutBuffer[y + 1];
        }
        delete [] realOutBuffer;
        delete [] imagOutBuffer;
    }
    // Reclaim some memory
    delete [] fft_real;
    delete [] fft_imag;
}

void cpu_ffty(float *real_image, float *imag_image, int size_x, int size_y)
{
    
    unsigned int x = 0;
    unsigned int y = 0;
    float *real_trans = new float[size_y * size_x];
    float *imag_trans = new float[size_y * size_x];
    for (x = 0; x < size_x; x++)
    {
        for (y = 0; y < size_y; y++)
        {
            real_trans[x*size_x + y] = real_image[y*size_x + x];
            imag_trans[x*size_x + y] = imag_image[y*size_x + x];
        }
    }

    cpu_fftx(real_trans, imag_trans, size_y, size_x);

    for (x = 0; x < size_x; x++)
    {
        for (y = 0; y < size_y; y++)
        {
            real_image[x*size_x + y] = real_trans[y*size_x + x];
            imag_image[x*size_x + y] = imag_trans[y*size_x + x];
        }
    }

    delete [] real_trans;
    delete [] imag_trans;
    return;
}

// This is the same as the thing about it, but it includes a scaling factor
void cpu_iffty(float *real_image, float *imag_image, int size_x, int size_y)
{
    unsigned int x = 0;
    unsigned int y = 0;
    float *real_trans = new float[size_y * size_x];
    float *imag_trans = new float[size_y * size_x];
    for (x = 0; x < size_x; x++)
    {
        for (y = 0; y < size_y; y++)
        {
            real_trans[x*size_x + y] = real_image[y*size_x + x];
            imag_trans[x*size_x + y] = imag_image[y*size_x + x];
        }
    }

    cpu_ifftx(real_trans, imag_trans, size_y, size_x);

    for (x = 0; x < size_x; x++)
    {
        for (y = 0; y < size_y; y++)
        {
            real_image[x*size_x + y] = real_trans[y*size_x + x];
            imag_image[x*size_x + y] = imag_trans[y*size_x + x];
        }
    }

    delete [] real_trans;
    delete [] imag_trans;
    return;
}

void cpu_filter(float *real_image, float *imag_image, int size_x, int size_y)
{
    int eightX = size_x/8;
    int eight7X = size_x - eightX;
    int eightY = size_y/8;
    int eight7Y = size_y - eightY;
    unsigned int x = 0;
    unsigned int y = 0;
    #pragma omp parallel for collapse(2) \
            shared(eightX, eight7X, eightY, eight7Y, real_image, imag_image) \
            private(x,y)
    for(x = 0; x < size_x; x++)
    {
        for(y = 0; y < size_y; y++)
        {
            if(!(x < eightX && y < eightY) &&
                    !(x < eightX && y >= eight7Y) &&
                    !(x >= eight7X && y < eightY) &&
                    !(x >= eight7X && y >= eight7Y))
            {
                // Zero out these values
                real_image[x*size_y + y] = 0;
                imag_image[x*size_y + y] = 0;
            }
        }
    }
}

float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y)
{
    // These are used for timing
    struct timeval tv1, tv2, tv3, tv4, tv5, tv6, tv7;
    struct timezone tz1, tz2, tz3, tz4, tz5, tz6, tz7;

    // Start timing
    gettimeofday(&tv1,&tz1);

    // Perform fft with respect to the x direction
    cpu_fftx(real_image, imag_image, size_x, size_y);
    //gettimeofday(&tv2,&tz2);

    // Perform fft with respect to the y direction
    cpu_ffty(real_image, imag_image, size_x, size_y);
    //gettimeofday(&tv3,&tz3);

    // Filter the transformed image
    cpu_filter(real_image, imag_image, size_x, size_y);
    //gettimeofday(&tv4,&tz4);

    // Perform an inverse fft with respect to the x direction
    cpu_ifftx(real_image, imag_image, size_x, size_y);
    //gettimeofday(&tv5,&tz5);

    // Perform an inverse fft with respect to the y direction
    cpu_iffty(real_image, imag_image, size_x, size_y);
    //gettimeofday(&tv6,&tz6);

    // End timing
    gettimeofday(&tv7,&tz7);

    // Compute the time difference in micro-seconds
    /*float execution1 = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
    execution1 /= 1000;
    float execution2 = ((tv3.tv_sec-tv2.tv_sec)*1000000+(tv3.tv_usec-tv2.tv_usec));
    execution2 /= 1000;
    float execution3 = ((tv4.tv_sec-tv3.tv_sec)*1000000+(tv4.tv_usec-tv3.tv_usec));
    execution3 /= 1000;
    float execution4 = ((tv5.tv_sec-tv4.tv_sec)*1000000+(tv5.tv_usec-tv4.tv_usec));
    execution4 /= 1000;
    float execution5 = ((tv6.tv_sec-tv5.tv_sec)*1000000+(tv6.tv_usec-tv5.tv_usec));
    execution5 /= 1000;*/
    float execution6 = ((tv7.tv_sec-tv1.tv_sec)*1000000+(tv7.tv_usec-tv1.tv_usec));
    execution6 /= 1000;
    // Print some output
    printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
    /*
    printf("  fftx Execution Time: %f ms\n\n", execution1);
    printf("  ffty Execution Time: %f ms\n\n", execution2);
    printf("  filter Execution Time: %f ms\n\n", execution3);
    printf("  ifftx Execution Time: %f ms\n\n", execution4);
    printf("  iffty Execution Time: %f ms\n\n", execution5);
    */
    printf("  Optimized Kernel Execution Time: %f ms\n\n", execution6);
    return execution6;
}
