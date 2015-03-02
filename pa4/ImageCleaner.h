#ifndef __IMAGE_CLEANER__
#define __IMAGE_CLEANER__

// This function contains the reference implementation of the algorithm needed
// to be implemented for OpenMP
// The float that is returned is the total time required to execute this kernel
float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y);

#endif
