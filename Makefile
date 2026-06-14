CC = gcc

CFLAGS = -O3 -march=native -ffast-math -funroll-loops -pthread
LDFLAGS = -lm -pthread

all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) matmul.c -o matmul $(LDFLAGS)

clean:
	rm -f matmul *.o

benchmark: matmul
	./matmul

.PHONY: all clean benchmark

