CC = gcc
CFLAGS = -fopenmp -O3

all: countri

countri: countri.o
	$(CC) -o countri countri.o $(CFLAGS) 

countri.o: countri.c
	$(CC) -c countri.c $(CFLAGS)

clean:
	rm *.o