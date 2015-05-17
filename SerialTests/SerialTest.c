#include <math.h>
#include <stdio.h>
#include <string.h>
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
     printf("Data dimensions - %d by %d\n", mnIn[0], mnIn[1]);
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
   double *R;
   if(NULL==(R = malloc(m*n*sizeof(double)))){
    printf("malloc of R failed\n");
    return(-1);
   }
   printf("R is allocated!\n");
   double *tau;
   if(NULL==(tau = malloc(n*sizeof(double)))){
     printf("malloc of Tau failed\n");
     return(-1);
   }
   printf("tau is allocated!\n");

   //Read in data file
   fid = fopen("TestIn.dat", "r");
   if(fid ==NULL){
     printf("TestIn.dat file could not open!\n");
     return(-1);
   }else{
     fread(A, sizeof(double), m*n, fid);
     printf("Reading in Data file of size %d by %d\n", mnIn[0], mnIn[1]);
     fclose(fid);
   }
   printf("Finished reading in the %d data points!\n", m*n);


   //Nice if we have smaller matrices
   //print_matrix_colmajor("Entry Matrix A", m, n, A, lda);
   //printf( "\n" );

   //Solving the Double GEneral matrix QR decomposition

   printf("Solving the QR decomp...");
   info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau);
   printf("done!\n");

   //Use this to make a smaller upper triangular matrix
   //int upperTriSize = n*(n+1)/2;
   //double R_upperTri[upperTriSize];
   
   //Copy over the data from the solution matrix A to the upper tri R
   printf("Memcpy results to R matrix...");
   memcpy(R, A, sizeof(double)*m*n);
   printf("done!\n");

   printf("Getting the Q data from the QGEQRF output matrix...");
   info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, A, lda, tau);
   printf("done!\n");

   //Lapacke svd function
   //Allocating Solution Matrices
   double S[n], U[m*m], Vt[n*n];
   double superb[n-1];
   info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, R, m, S, U, m, Vt, n, superb);

   //TODO: Write function to compare with MATLAB solutions to do error checking

   //TODO: Need to deallocate solution matrices

   return(0);
}

int allocateData(double *dataPtr, int length, char* description){
   if(NULL==(dataPtr = malloc(length*sizeof(double)))){
     printf("malloc of %s failed\n", description);
     return(-1);
   }
   printf("%s is allocated!\n", description);
   return(0);
}  
