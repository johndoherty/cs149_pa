#include "CpuReference.h"
#include "ImageCleaner.h"
#include "JPEGWriter.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string>

int main(int argc, char **argv)
{
  if(argc != 2)
  {
    fprintf(stderr,"Usage: ImageCleaner noisy_image_file.nsy\n");
    exit(1);
  }

  // Set up to make sure that we are running on the right GPU on the
  // Amazon EC2 cluster. If you want to run on your own GPU you will
  // have to modify this code.
  {
    int numDevices;
    CUDA_ERROR_CHECK(cudaGetDeviceCount(&numDevices));

    // Set the device to be the first one that does not have a timeout enabled
    int devicesConsidered=0;
    for ( ; devicesConsidered<numDevices; devicesConsidered++)
    {
      cudaDeviceProp prop;
      CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop,devicesConsidered));
      if (!(prop.kernelExecTimeoutEnabled))
      {
        fprintf(stdout,"Selecting GPU %d of type %s\n\n",devicesConsidered,prop.name);
        CUDA_ERROR_CHECK(cudaSetDevice(devicesConsidered));
        break;
      }
    }
    // If we make it here and couldn't find a device without a timeout
    // enabled, exit.
    if (devicesConsidered == numDevices)
    {
      fprintf(stderr,"ERROR: Unable to find a GPU in compute mode!\n");
      exit(1);
    }
  }

  // Read in the noisy image file
  char *fileName = argv[1];
  std::ifstream ifs;
  ifs.open(fileName, std::ios::in);
  
  int runRef = 0;

  // Get the size of the image
  int size_x, size_y;
  ifs >> size_x >> size_y;
  assert((size_x > 0) && (size_y > 0));
  
  // Allocate space for the real and imaginary components of the image
  float *real_image = new float[size_x * size_y];
  float *imag_image = new float[size_x * size_y];
  // Also allocate space for the reference implementation
  float *real_image_ref = new float[size_x * size_y];
  float *imag_image_ref = new float[size_x * size_y];

  // Read in the components
  for(unsigned int i=0; i< size_x; i++)
  {
    for(unsigned int j=0; j<size_y; j++)
    {
      float real, imag;
      // Read in both the real and imaginary parts
      ifs >> real >> imag;
      // Write them into the arrays
      real_image[i*size_x + j] = real;
      imag_image[i*size_x + j] = imag;
      // Do the same for the reference
      real_image_ref[i*size_x + j] = real;
      imag_image_ref[i*size_x + j] = imag;
    }
  }
  // Close the file since we're done with it now
  ifs.close();

  // Call the functions to perform both the reference and the CUDA implementations
  float referenceTime = 0.0f;
  if (runRef) {
    referenceTime = referenceCleaner(real_image_ref, imag_image_ref, size_x, size_y);
  }
  else {
    if (size_x == 512) {
      referenceTime = 57499.5f;
    }
    else if (size_x == 1024) {
      referenceTime = 484880.0f;
    }
  }

  // Call the GPU implementation
  float cudaTime = filterImage(real_image, imag_image, size_x, size_y);

  // Print out the speedup statistic
  printf("TOTAL SPEEDUP: %f\n\n", (referenceTime/cudaTime));

  // Dump the image to jpeg format for the cuda implementation
  {
    std::string tempFileName(fileName);
    // Take off the '.nsy' postfix and post-pend '_out.jpg'
    int index = tempFileName.find(".nsy");
    // Better have found it
    assert(index >= 0);
    std::string outFileName = tempFileName.substr(0,index);
    outFileName.append("_out.jpg");
    printf("Writing out CUDA generated image to %s\n\n", outFileName.c_str());
    write_jpeg(outFileName.c_str(), real_image, imag_image, size_x, size_y); 
  }
  // Clean up the memory
  delete [] real_image;
  delete [] imag_image;
  delete [] real_image_ref;
  delete [] imag_image_ref; 

  return 0; 
}
