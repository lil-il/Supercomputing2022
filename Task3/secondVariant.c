#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double* mul(double *matrix, double *vector, double *result, int cols, int rows, int sh, int N){
    int i,j;
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            result[i] += matrix[i*N + (j + sh)]*vector[j];
        }
    }
    return result;
};

double* sub(double *vec1, double *vec2, double *result, int N){
    int i;
    for(i = 0; i < N; i++){
        result[i] = vec1[i] - vec2[i];
    }
    return result;
};

double* scmul(double *vec, double tau, int N){
    int i;
    for(i = 0; i < N; i++) {
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
    double *b = (double*)malloc((N/size + shift)*sizeof(double));
    double *S = (double*)malloc((N/size + shift)*sizeof(double));
    double *R = (double*)malloc((N/size + shift)*sizeof(double));
    double *x = (double*)malloc((N/size + 1)*sizeof(double));
    double *x_ = (double*)malloc((N/size + 1)*sizeof(double));

    int rows = 0;
    int i, j;

    for(i = 0; i < rank; i++){
        rows += N/size + (i < N % size ? 1 : 0);
    }

    for(i = 0; i < N/size + shift; i++){
        for(j = 0; j < N; j++){
            if((i*N + j - rows) % (N + 1) == 0){
                a[i*N + j] = 2.0;
            }
            else a[i*N + j] = 1.0;
        }
    }
    double norm_B = 0;
    double nB;
    for(i = 0; i < N/size + shift; i++){
        b[i] = N + 1;
        norm_B += b[i]*b[i];
    }

    MPI_Allreduce(&norm_B, &nB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memset(x, 0, (N/size + 1)*sizeof(double));

    /*____________________________________________________________________________________*/
    double t1 = MPI_Wtime();

    double norm_U;
    double c;

    int shift_ = 0;
    int rank_ = 0;
    double *q;
    /*____________________________________________________________________________________*/
    int p;

    for(;;){
        norm_U = 0;
        memset(R, 0, (N/size + shift)*sizeof(double));
        memset(S, 0, (N/size + shift)*sizeof(double));

        shift_ = shift;

        for(p = 0; p < size; p++){
            mul(a, x, R,   N/size + shift_, N/size + shift, rows, N);
            MPI_Sendrecv(x, N/size + 1, MPI_DOUBLE, (rank + 1) % size, 0, x_, N/size + 1, MPI_DOUBLE, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, 0);
            q = x;
            x = x_;
            x_ = q;
            shift_ = ((rank - 1 - p + size) % size < N % size) ? 1 : 0;
            rank_ = (rank - 1 - p + size) % size;
            rows = 0;

            for(i = 0; i < rank_; i++){
                rows += N/size + (i < N % size ? 1 : 0);
            }
        }

        sub(R, b, S, N/size + shift);

        for(i = 0; i < N/size + shift; i++){
            norm_U += S[i]*S[i];
        }
        MPI_Allreduce(&norm_U, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        scmul(S, tau_p, N/size + shift);
        sub(x, S, R, N/size + shift);
        // memcpy(x, R, (N/size + shift)*sizeof(double));
        for(i = 0; i < (N/size +shift); i++){
            x[i] = R[i];
        };

        if(c/nB < EPS*EPS){
            int count = 0;
            for(i = 0; i < N/size + shift; i++){
                // printf("%d %.10f\n", rank, x[i]);
                if(fabs(x[i] - 1) < EPS) continue;
                else {
                    ++count;
                    break;
                }
            }
            printf("%d\n", count);
            break;
        }
    }
    /*____________________________________________________________________________________*/

    double t2 = MPI_Wtime();
    double t3 = t2 - t1;
    double t4;

    MPI_Reduce(&t3, &t4, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("N = %d\nThreads = %d\nTime = %f\n", N, size, t4);
    }
    // printf("%d %f\n", rank, t2 - t1);

    MPI_Finalize();

    free(a);
    free(b);
    free(R);
    free(S);
    free(x);
    free(x_);

    return 0;
}