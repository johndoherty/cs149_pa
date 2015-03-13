DEBUG ?= 1

SIZEX ?= 1024
SIZEY ?= 1024

OBJECTS = main.o ImageCleaner.o JPEGWriter.o CpuReference.o

NVCCFLAGS = -arch=compute_30 -code=sm_30 -Xptxas "-v" -D SIZEX=$(SIZEX) -D SIZEY=$(SIZEY)
ifeq ($(DEBUG),1)
NVCCFLAGS += -g -G
else
NVCCFLAGS +=
endif
NVCCLDFLAGS = -L /usr/local/cuda/lib64
NVCCLIBS = -lcudart -ljpeg

CFLAGS = -I /usr/local/cuda/include -D SIZEX=$(SIZEX) -D SIZEY=$(SIZEY)
ifeq ($(DEBUG),1)
CFLAGS += -g
else
CFLAGS +=
endif

.PHONY: all
all: $(OBJECTS)
	nvcc $(NVCCFLAGS) $(NVCCLDFLAGS) -o ImageCleaner $(OBJECTS) $(NVCCLIBS)

main.o: main.cc
	g++ $(CFLAGS) -c -o main.o main.cc

ImageCleaner.o: ImageCleaner.cu
	nvcc $(NVCCFLAGS) -c -o ImageCleaner.o ImageCleaner.cu

JPEGWriter.o: JPEGWriter.cc
	g++ $(CFLAGS) -c -o JPEGWriter.o JPEGWriter.cc

CpuReference.o: CpuReference.cc
	g++ $(CFLAGS) -c -o CpuReference.o CpuReference.cc

.PHONY: clean
clean:
	rm -f *~ *.o *.linkinfo ImageCleaner
