#include <math.h>
#include <stdio.h>
#include <lapacke.h>
#include "lapacke_example_aux.h"

int main(int argc, char *argv[])
{

   double A[5][3] = {1.3234,2.13412,.1239421,2.2341,3.123431,4.341,2.2342,5.12,2.342,4.543,.1234,5.23593,0.2134,4.1234,3.432};
   double R[5][3];
   double tau[5];
   lapack_int info, m, n, lda;
   
   m = 5;
   n = 3;
   lda = 5;

   int i, j;

   print_matrix_colmajor("Entry Matrix A", m, n, *A, lda);
   printf( "\n" );

   info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, *A, lda, tau);

   int upperTriSize = n*(n+1)/2;
   double R_upperTri[upperTriSize];
   
   memcpy(*R, *A, sizeof(double)*m*n);

   print_matrix_colmajor("R Solution", m, n, *A, lda);
   printf( "\n" );

   print_matrix_colmajor("R Solution Copy", m, n, *R, lda);
   printf( "\n" );

   info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, *A, lda, tau);
   print_matrix_colmajor("Q Solution", m, n, *A, lda);
   printf( "\n" );

   //Testing out the lapacke svd function
   
   double S[n], U[m*m], Vt[n*n];
   double superb[n-1];
   info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, *R, m, S, U, m, Vt, n, superb);

   print_matrix_colmajor("U", m, m, U, lda);
   printf( "\n" );

   print_matrix_colmajor("S", n, 1, S, 1);
   printf( "\n" );

   print_matrix_colmajor("Vt", n, n, Vt, n);
   printf( "\n" );


   exit(0);
}  
