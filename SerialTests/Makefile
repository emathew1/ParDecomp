LOADER = gfortran
LOPT = -O3 
CC = gcc
MAIN = SerialTest
HEAD = lapacke_example_aux
CCOPT  = -O3

all: $(MAIN)

LIBRARIES  = /Users/edwin/Downloads/lapack-3.5.0/liblapacke.a /Users/edwin/Downloads/lapack-3.5.0/liblapack.a /Users/edwin/Downloads/lapack-3.5.0/liblapacke.a /Users/edwin/Downloads/lapack-3.5.0/librefblas.a

$(MAIN):  $(MAIN).o $(HEAD).o $(LIBRARIES)
	$(LOADER) $(LOPT) $(MAIN).o $(HEAD).o $(LIBRARIES) -o $@

.c.o:
	$(CC) -c $(CCOPT) -I. -I /Users/edwin/Downloads/lapack-3.5.0/lapacke/include -o $@ $<

clean:
	rm *.o $(MAIN)
