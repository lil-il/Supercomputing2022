#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double* mul(double *matrix, double *vector, double *result, int N, int rows){
    int i,j;
    for( i = 0; i < rows; i++){
        for( j = 0; j < N; j++){
            result[i] += matrix[i*N + j]*vector[j];
        }
    }
    return result;
};

double* sub(double *vec1, double *vec2, double *result, int N){
    int i;
    for( i = 0; i < N; i++){
        result[i] = vec1[i] - vec2[i];
    }
    return result;
};

double* scmul(double *vec, double tau, int N){
    int i;
    for ( i = 0; i < N; i++) {
        vec[i] *= tau;
    }
    return vec;
};

const double EPS = 10e-9;
const double tau_p = 10e-6;
const double tau_n = -0.001;

int main(int argc, char* argv[]){
    int N = 5000;
    int rank, size;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    if(size > N){
        if(rank == 0) printf("Too many threads!\n");
        MPI_Finalize();
        return 0;
    }
    /*____________________________________________________________________________________*/

    int shift = ((rank < N % size) ? 1 : 0);

    double *a = (double*)malloc(N*(N/size + shift)*sizeof(double));
    double *b = (double*)malloc((N)*sizeof(double));
    double *S = (double*)malloc((N/size + shift)*sizeof(double));
    double *R = (double*)malloc((N/size + shift)*sizeof(double));
    double *x = (double*)malloc(N*sizeof(double));

    int rows = 0;
    int i;
    int j;
    for( i = 0; i < rank; i++){
        rows += N/size + (i < N % size ? 1 : 0);
    }
    for( i = 0; i < N/size + shift; i++){
        for( j = 0; j < N; j++){
            if((i*N + j - rows) % (N + 1) == 0){
                a[i*N + j] = 2.0;
            }
            else a[i*N + j] = 1.0;
        }
    }
    double norm_B = 0;
    for( i = 0; i < N; i++){
        b[i] = N + 1;
        norm_B += b[i]*b[i];
    }
    memset(x, 0, N*sizeof(double));
    /*____________________________________________________________________________________*/

    double t1 = MPI_Wtime();

    /*____________________________________________________________________________________*/
    double norm_U;
    double c;

    int *recvcounts = (int*)malloc(size*sizeof(int));
    memset(recvcounts, 0, size*sizeof(int));
    for( i = 0; i < size; i++){
        recvcounts[i] = N/size + (i < N % size ? 1 : 0);
    }

    int *displs = (int*)malloc(size*sizeof(int));
    memset(displs, 0, size*sizeof(int));
    int y = 0;
    for( i = 1; i < size; i++){
        y += N/size + ((i - 1) < N % size ? 1 : 0);
        displs[i] = y;
    }

    for(;;){
        memset(R, 0, (N/size + shift)*sizeof(double));
        memset(S, 0, (N/size + shift)*sizeof(double));

        norm_U = 0;
        mul(a, x, R, N, N/size + shift);

        sub(R, b + rows, S, N/size + shift);
        for( i = 0; i < N/size + shift; i++){
            norm_U += S[i]*S[i];
        }
        MPI_Allreduce(&norm_U, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        scmul(S, tau_p, N/size + shift);
        sub(x + rows, S, R, N/size + shift);
        // memcpy(x + rows, R, (N/size + shift)*sizeof(double));
        for(i = 0; i < N/size + shift; i++){
            x[i + rows] = R[i];
        }

        if(c/norm_B < EPS*EPS){
            int count = 0;
            for( i = 0; i < N/size + shift; i++){
                // printf("%d %.10f\n", rank, x[rows + i]);
                if(fabs(x[rows + i] - 1) < EPS) continue;
                else {
                    ++count;
                    break;
                }
            }
            if (count > 0) printf("%d\n", count);
            break;
        }

        MPI_Allgatherv(R, N/size + shift, MPI_DOUBLE, x, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    /*____________________________________________________________________________________*/

    double t2 = MPI_Wtime();
    double t3 = t2 - t1;
    double t4;

    MPI_Allreduce(&t3, &t4, 1, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD);
    if(rank == 0){
        printf("N = %d\nThreads = %d\nTime = %f\n", N, size, t4);
    }

    free(a);
    free(b);
    free(R);
    free(S);
    free(x);
    free(displs);
    free(recvcounts);
    MPI_Finalize();
    return 0;
}