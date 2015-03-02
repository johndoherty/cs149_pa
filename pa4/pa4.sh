#!/bin/sh

# Set number of OpenMP threads
export OMP_NUM_THREADS=8

# Clean up the directory
make clean

# Compile the program
make DEBUG=0

# Run the program
./ImageCleaner images/noisy_01.nsy
./ImageCleaner images/noisy_02.nsy
./ImageCleaner images/noisy_03.nsy
