# This is for use ONLY with the competition portion of the extra
# credit. For all other parts of the assignment, you must use the
# regular Makefile (including the non-competition portions of the
# extra credit). To use, invoke:
#
# make -f Competition.Makefile DEBUG=0

DEBUG ?= 1

SOURCES = main.cc JPEGWriter.cc CpuReference.cc ImageCleaner.cc
LIBS = -ljpeg

CFLAGS = -fopenmp
ifeq ($(DEBUG),1)
CFLAGS += -g
else
CFLAGS += -O3
endif

.PHONY: all
all:
	g++ $(CFLAGS) -o ImageCleaner $(SOURCES) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o ImageCleaner
