DEBUG ?= 1

SOURCES = main.cc JPEGWriter.cc CpuReference.cc ImageCleaner.cc
LIBS = -ljpeg

CFLAGS = -fopenmp
ifeq ($(DEBUG),1)
CFLAGS += -g
else
CFLAGS +=
endif

.PHONY: all
all:
	g++ $(CFLAGS) -o ImageCleaner $(SOURCES) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o ImageCleaner
