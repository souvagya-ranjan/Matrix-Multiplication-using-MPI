//P2P non-blocking communication
#include <bits/stdc++.h>
#include <mpi.h>
#include <chrono>

using namespace std;

void multiply_seriel(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float sum = 0;
            for(int k = 0; k < 32; k++){
                sum += a[i*32 + k] * b[k*n + j];
            }
            c[i*n + j] = sum;
        }
    }
}

void print_matrix(float *a, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            cout << a[i*m + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

bool IsEqual(float *a, float *b, int n){
    for(int i = 0; i < n*n; i++){
        if(abs(a[i] - b[i]) > 0.001){
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]){
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request request;
    MPI_Status status;

    int start_index, end_index, n;
    n = stoi(argv[1]);
    chrono::high_resolution_clock::time_point start_time, end_time;

    if(rank == 0){
        float *a = new float[n*32];
        for(int i = 0; i < n; i++){
            for(int j = 0; j < 32; j++){
                //random number between 0 and 1 inclusive
                a[i*32 + j] = (float)rand() / RAND_MAX; 
            }
        }
        float *b = new float[32*n];
        for(int i = 0; i < 32; i++){
            for(int j = 0; j < n; j++){
                b[i*n + j] = (float)rand() / RAND_MAX;
            }
        }
        float *c_seriel = new float[n*n];
        multiply_seriel(a, b, c_seriel, n);
        float *c_parallel = new float[n*n];
        start_time = chrono::high_resolution_clock::now();
        for(int i = 1; i < size; i++){
            MPI_Isend(b, 32*n, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request);
        }
        MPI_Wait(&request, &status);

        for(int i = 1; i < size; i++){
            start_index = (i-1) * n / (size-1);
            end_index = i * n / (size-1);
            if(i == size-1){
                end_index = n;
            }
            MPI_Isend(&a[start_index*32], (end_index - start_index)*32, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request);
        }
        MPI_Wait(&request, &status);
        for(int i = 1; i < size; i++){
            start_index = (i-1) * n / (size-1);
            end_index = i * n / (size-1);
            if(i == size-1){
                end_index = n;
            }
            MPI_Irecv(&c_parallel[start_index*n], (end_index - start_index)*n, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request);
        }
        MPI_Wait(&request, &status);
        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;
        double seconds = elapsed.count();
        cout<< "Time taken by the code3 is: "<<seconds<<" nano-seconds"<<endl;
        if(IsEqual(c_seriel, c_parallel, n)){
            cout << "Correct result" << endl;
        }
        else{
            cout << "Incorrect result" << endl;
        }
    }
    else{
        float *b_worker = new float[32*n];
        MPI_Irecv(b_worker, 32*n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        start_index = (rank-1) * n / (size-1);
        end_index = rank * n / (size-1);
        if(rank == size-1){
            end_index = n;
        }
        float *a_worker = new float[(end_index - start_index)*32];
        MPI_Irecv(a_worker, (end_index - start_index)*32, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);

        MPI_Wait(&request, &status);

        float *c_worker = new float[(end_index - start_index)*n];
        for(int i = 0; i < end_index - start_index; i++){
            for(int j = 0; j < n; j++){
                c_worker[i*n + j] = 0;
                for(int k = 0; k < 32; k++){
                    c_worker[i*n + j] += a_worker[i*32 + k] * b_worker[k*n + j];
                }
            }
        }
        MPI_Isend(c_worker, (end_index - start_index)*n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);    
    }
    MPI_Finalize();
    return 0;
}