CFLAGS=-g -Wall -std=c99 -O3 -march=native
LDLIBS=-lm

all: fft convoluzione
conv_fft: fft.h
