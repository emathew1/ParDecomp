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
   lda = n;

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


   //Solving the Double GEneral matrix QR decomposition
   printf("Solving the QR decomp...");
   info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, lda, tau);
   printf("done!\n"); 

   //Use this to make a smaller upper triangular matrix
   //int upperTriSize = n*(n+1)/2;
   //double R_upperTri[upperTriSize];
   
   //Copy over the data from the solution matrix A to the upper tri R
   printf("Memcpy results to R matrix...");
   memcpy(R, A, sizeof(double)*m*n);
   printf("done!\n");

   

   printf("Getting the Q data from the QGEQRF output matrix...");
   info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, n, A, lda, tau);
   printf("done!\n");

   //Lapacke svd function
   //Allocating Solution Matrices
   double *S, *U, *Vt, *superb;

   //Allocating S
   if(NULL==(S = malloc(n*sizeof(double)))){
     printf("malloc of S failed\n"); return(-1);
   }else{ printf("S is allocated!\n");}

   //Allocating U
   if(NULL==(U = malloc(m*m*sizeof(double)))){
     printf("malloc of U failed\n");return(-1);
   }else{printf("U is allocated!\n");}

   //Allocating Vt
   if(NULL==(Vt = malloc(n*n*sizeof(double)))){
     printf("malloc of Vt failed\n");return(-1);
   }else{printf("Vt is allocated!\n");}
   
   //Allocating superb
   if(NULL==(superb = malloc((n-1)*sizeof(double)))){
     printf("malloc of supurb failed\n");return(-1);
   }else{printf("superb is allocated!\n");}



   printf("Calculating the SVD of Matrix R using DGESVD...");
   info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, R, lda, S, U, m, Vt, n, superb);
   printf("done!\n");

   //print_matrix_rowmajor("U", m, m, Vt, m);
   //print_matrix_rowmajor("V", n, n, Vt, n);
   //print_matrix_rowmajor("S", 1, n, S, 1);

   //TODO: Write function to compare with MATLAB solutions to do error checking

   //TODO: Need to deallocate solution matrices


   return(0);
}
