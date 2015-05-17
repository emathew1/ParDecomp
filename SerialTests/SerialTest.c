#include <math.h>
#include <stdio.h>
#include <lapacke.h>
#include "lapacke_example_aux.h"

int main(int argc, char *argv[])
{

   //Read in data size file
   FILE *fid; int mnIn[2];
   fid = fopen("TestSize.dat", "r");
   if(fid ==NULL){
     printf("TestSize.dat file could not open!\n");
     return(-1);
   }else{
     fread(mnIn, sizeof(int), 2, fid);
     printf("Reading in Data file of size %d by %d\n", mnIn[0], mnIn[1]);
     fclose(fid);
   }
  
   //set up lapack variables
   lapack_int info, m, n, lda;
   m = mnIn[0];
   n = mnIn[1];
   lda = m;


   //Allocate room for data
   double *A;
   if(NULL==(A = malloc(m*n*sizeof(double)))){
     printf("malloc of A failed\n");
     return(-1);
   }
   printf("A is allocated!\n");
   double R[m*n];
   double tau[n];


   //Nice if we have smaller matrices
   //print_matrix_colmajor("Entry Matrix A", m, n, A, lda);
   //printf( "\n" );

   //Solving the Double GEneral matrix QR decomposition
   info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau);

   //Use this to make a smaller upper triangular matrix
   int upperTriSize = n*(n+1)/2;
   double R_upperTri[upperTriSize];
   
   //Copy over the data from the solution matrix A to the upper tri R
   memcpy(R, A, sizeof(double)*m*n);


   //Nice if we have smaller matrices
   //print_matrix_colmajor("R Solution", m, n, A, lda);
   //printf( "\n" );

   //Nice if we have smaller matrices
   //print_matrix_colmajor("R Solution Copy", m, n, R, lda);
   //printf( "\n" );

   //This will give us the Q of the QR decomp from the previous Lapacke function
   info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, A, lda, tau);
   //print_matrix_colmajor("Q Solution", m, n, A, lda);
   //printf( "\n" );

   //Lapacke svd function

   //Allocating Solution Matrices
   double S[n], U[m*m], Vt[n*n];
   double superb[n-1];
   info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, R, m, S, U, m, Vt, n, superb);

   //print_matrix_colmajor("U", m, m, U, lda);
   //printf( "\n" );

   //print_matrix_colmajor("S", n, 1, S, 1);
   //printf( "\n" );

   //print_matrix_colmajor("Vt", n, n, Vt, n);
   //printf( "\n" );

   //Write function to compare with MATLAB solutions to do error checking

   //Need to deallocate solution matrices

   exit(0);
}  
