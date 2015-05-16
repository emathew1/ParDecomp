#include <math.h>
#include <stdio.h>
#include <lapacke.h>
#include "lapacke_example_aux.h"

int main(int argc, char *argv[])
{


   lapack_int info, m, n, lda;
   m = 5;
   n = 3;
   lda = 5;

   double A[m*n];
   A[0] = 1.3234;
   A[1] = 2.13412;
   A[2] = .1239421;
   A[3] = 2.2341;
   A[4] = 3.123431; 
   A[5] = 4.341;
   A[6] = 2.2342;
   A[7] = 5.12;
   A[8] = 2.342;
   A[9] = 4.543;
   A[10]= .1234;
   A[11]= 5.23593;
   A[12]= 0.2134;
   A[13] = 4.1234;
   A[14] =3.432;
   double R[m*n];
   double tau[n];
   

   int i, j;

   print_matrix_colmajor("Entry Matrix A", m, n, A, lda);
   printf( "\n" );

   info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau);

   int upperTriSize = n*(n+1)/2;
   double R_upperTri[upperTriSize];
   
   memcpy(R, A, sizeof(double)*m*n);

   print_matrix_colmajor("R Solution", m, n, A, lda);
   printf( "\n" );

   print_matrix_colmajor("R Solution Copy", m, n, R, lda);
   printf( "\n" );

   info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, A, lda, tau);
   print_matrix_colmajor("Q Solution", m, n, A, lda);
   printf( "\n" );

   //Testing out the lapacke svd function
   
   double S[n], U[m*m], Vt[n*n];
   double superb[n-1];
   info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, R, m, S, U, m, Vt, n, superb);

   print_matrix_colmajor("U", m, m, U, lda);
   printf( "\n" );

   print_matrix_colmajor("S", n, 1, S, 1);
   printf( "\n" );

   print_matrix_colmajor("Vt", n, n, Vt, n);
   printf( "\n" );


   exit(0);
}  
