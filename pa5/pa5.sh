#!/bin/sh

# Compile
make DEBUG=0 SIZEX=512 SIZEY=512 clean all

# Run the smaller images
./ImageCleaner images/noisy_01.nsy
./ImageCleaner images/noisy_02.nsy

# Compile
make DEBUG=0 SIZEX=1024 SIZEY=1024 clean all

# Run the larger images
./ImageCleaner images/noisy_03.nsy
