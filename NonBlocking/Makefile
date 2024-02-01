CC=mpic++ 
CFLAGS=-std=c++11 -g -lmpi -O2

exec: Collective.cpp
	$(CC) $(CFLAGS) NonBlocking.cpp -o exec

run: Collective.cpp
	mpirun -np 4 ./exec 1000
clean:
	rm -rf *.o exec
