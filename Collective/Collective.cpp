//MPI collective communication
#include <bits/stdc++.h>
#include <mpi.h>
#include <chrono>

using namespace std;

void multiply_seriel(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float sum = 0;
            for(int k = 0; k < 6; k++){
                sum += a[i*6 + k] * b[k*n + j];
            }
            c[i*n + j] = sum;
        }
    }
}

bool IsEqual(float *a, float *b, int n){
    for(int i = 0; i < n*n; i++){
        if(abs(a[i] - b[i]) > 0.0001){
            return false;
        }
    }
    return true;
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

int main(int argc, char* argv[]){
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_index, end_index, n;
    chrono::high_resolution_clock::time_point start_time, end_time;

    n = stoi(argv[1]);
    float *a, *a_worker, *c_worker, *c_parallel, *c_seriel;
    float *b = new float[6*n];
    if(rank == 0){
        a = new float[n*6];
        for(int i = 0; i < n; i++){
            for(int j = 0; j < 6; j++){
                //random number between 0 and 1 inclusive
                a[i*6 + j] = (float)rand() / RAND_MAX; 
            }
        }
        b = new float[6*n];
        for(int i = 0; i < 6; i++){
            for(int j = 0; j < n; j++){
                b[i*n + j] = (float)rand() / RAND_MAX;
            }
        }
        c_seriel = new float[n*n];
        // start_time = chrono::high_resolution_clock::now();
        multiply_seriel(a, b, c_seriel, n);
        // end_time = chrono::high_resolution_clock::now();
        // chrono::duration<double> time_elapsed = end_time - start_time;
        // double seconds = time_elapsed.count();
        // cout << "Time taken by seriel code: " << seconds << " nano-seconds" << endl;
        c_parallel = new float[n*n];
    }
    start_time = chrono::high_resolution_clock::now();
    MPI_Bcast(b, 6*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // cout<<rank<<endl;
    // print_matrix(b, 6, n);
    
    int sendcounts[size];
    int displs[size];
    sendcounts[0] = 0;
    displs[0] = 0;
    for(int i = 1; i < size; i++){
        start_index = (i-1) * n / (size-1);
        end_index = i * n / (size-1);
        if(i == size-1){
            end_index = n;
        }
        sendcounts[i] = (end_index - start_index)*6;
        displs[i] = start_index*6;
    }
    int recvcount = sendcounts[rank];
    a_worker = new float[recvcount];
    MPI_Scatterv(a, sendcounts, displs, MPI_FLOAT, a_worker, recvcount, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(rank != 0){
        c_worker = new float[(sendcounts[rank]/6)*n];
        for(int i = 0; i < sendcounts[rank]/6; i++){
            for(int j = 0; j < n; j++){
                c_worker[i*n + j] = 0;
                for(int k = 0; k < 6; k++){
                    c_worker[i*n + j] += a_worker[i*6 + k] * b[k*n + j];
                }
            }
        }

    }
    int recvcounts_2[size];
    int displs_2[size]; 
    recvcounts_2[0] = 0;
    displs_2[0] = 0;
    for(int i = 1; i < size; i++){
        start_index = (i-1) * n / (size-1);
        end_index = i * n / (size-1);
        if(i == size-1){
            end_index = n;
        }
        recvcounts_2[i] = (end_index - start_index)*n;
        displs_2[i] = start_index*n;
    }

    MPI_Gatherv(c_worker, recvcounts_2[rank], MPI_FLOAT, c_parallel, recvcounts_2, displs_2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> time_elapsed = end_time - start_time;
    double seconds = time_elapsed.count();
    if(rank == 0){
        cout << "Time taken by parallel code: " << seconds << " nano-seconds" << endl;
    }        
    if(rank == 0){
        if(IsEqual(c_seriel, c_parallel, n)){
            cout << "Correct result" << endl;
            // print_matrix(c_seriel, n, n);
            // cout<<endl;
            // print_matrix(c_parallel, n, n);
        }
        else{
            cout << "Incorrect result" << endl;
        }
    }
    MPI_Finalize();
    return 0;
}
