#include <math.h>
#include <stdio.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
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
     printf(" ->Data dimensions - %d by %d\n", mnIn[0], mnIn[1]);
     fclose(fid);
   }
  
   //set up lapack variables
   lapack_int info, m, n, lda;
   m = mnIn[0];
   n = mnIn[1];
   lda = n;
 
   //Set mSmall Size (perfectly divisible size right now)
   int numOfChunks = 10;
   lapack_int mSmall = m/numOfChunks;
   

   //Allocate room for data
   double *A;
   if(NULL==(A = malloc(m*n*sizeof(double)))){
     printf("malloc of A failed\n");
     return(-1);
   }
   printf(" ->A is allocated!\n");
   double *R;
   if(NULL==(R = malloc(n*n*numOfChunks*sizeof(double)))){
    printf("malloc of R failed\n");
    return(-1);
   }
   printf(" ->R is allocated!\n");
   double *Tau;
   if(NULL==(Tau = malloc(numOfChunks*n*sizeof(double)))){
     printf("malloc of Tau failed\n");
     return(-1);
   }
   printf(" ->Tau is allocated!\n");

   //Read in data file
   fid = fopen("TestIn.dat", "r");
   if(fid ==NULL){
     printf("TestIn.dat file could not open!\n");
     return(-1);
   }else{
     fread(A, sizeof(double), m*n, fid);
     printf(" ->Reading in Data file of size %d by %d\n", mnIn[0], mnIn[1]);
     fclose(fid);
   }
   printf(" ->Finished reading in the %d data points!\n", m*n);

   //Solving the Double GEneral matrix QR decomposition
   int i;
   for(i = 0; i < numOfChunks; i++){
       printf(" ->Solving the QR decomp...%d...",i+1);
       //Need to move the pointer to the start of the next chunk after every
       //new index
       double *AchunkLocation = &A[i*mSmall*n]; 
       double *TauChunkLocation = &Tau[i*n]; 
       info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, mSmall, n, AchunkLocation, lda, TauChunkLocation);
       printf("done!\n"); 
   }
   
   //Lets see if we can be fairly organized about copying over just the upper tri's
   //of each chunk over to R 
   for(i = 0; i < numOfChunks; i++){
     double *AchunkLocation = &A[i*mSmall*n];
     double *RchunkLocation = &R[i*n*n];
     int j, k;
     for(j = 0; j < n; j++){
       for(k = 0; k < n; k++){
         if(k>=j){
           RchunkLocation[j*n + k] = AchunkLocation[j*n + k];
         }else{
           RchunkLocation[j*n + k] = 0.0;
	 }
       }
     }
   }
 
   //1st Lapack QR decomp function 
   for(i = 0; i < numOfChunks; i++){
     printf(" ->Getting the Q data from the QGEQRF output matrix...%d...", i+1);
     double *AchunkLocation = &A[i*mSmall*n];
     double *TauChunkLocation = &Tau[i*n]; 
     info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, mSmall, n, n, AchunkLocation, lda, TauChunkLocation);
     printf("done!\n");
   }
   free(Tau);


   //Need to do one more QR decomp of the upper-Tri block R Matrix
   double *Rfinal;
   if(NULL==(Rfinal = malloc(n*n*sizeof(double)))){
     printf("malloc of Rfinal failed\n");
     return(-1);
   }
   double *Tau2;
   if(NULL==(Tau2 = malloc(n*sizeof(double)))){
     printf("malloc of Tau2 failed\n");
     return(-1);
   }

   //2nd Lapack QR decomp function call 
   info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, numOfChunks*n, n, R, lda, Tau2);

   //Generate the final R matrix
   { //Scope the j and k iterators
     int j, k;
     for(j = 0; j < n; j++){
       for( k = 0; k < n; k++){
	 if(k >= j){
	   Rfinal[j*n + k] = R[j*n + k];
	 }else{
	   Rfinal[j*n + k] = 0.0;	
	 }
       }
     }
   }

   //Get the Q matrix back out of the last calculation
   //  For those keeping track, the Q matrices from the first QR decomp are now
   //  in A, and the Q matrix from the second decomp are now in R
   info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, numOfChunks*n, n, n, R, lda,Tau2);
   free(Tau2);

   //Start getting the real Q matrix back...
   double *Qfinal;
   if(NULL==(Qfinal = malloc(m*n*sizeof(double)))){
     printf("malloc of Qfinal failed\n");
     return(-1);
   } printf(" ->Allocated Qfinal!\n");

   for(i = 0; i < numOfChunks; i++){
     printf(" ->Calculating the final Q using DGEMM...%d...", i+1);
     double *AchunkLocation = &A[i*mSmall*n];
     double *RchunkLocation = &R[i*n*n];
     double *QchunkLocation = &Qfinal[i*mSmall*n];
     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mSmall, n, n, 1.0, AchunkLocation, n, RchunkLocation, n, 0.0, QchunkLocation, n); 	
     printf("done!\n");
   }
   free(R);
   free(A);  

   //print_matrix_rowmajor("R Final", n, n, Rfinal, n);
   //print_matrix_rowmajor("Q Final", m, n, Qfinal, n);

   //Lapacke svd function
   //Allocating Solution Matrices
   double *S, *U, *Vt, *superb;

   //Allocating S
   if(NULL==(S = malloc(n*sizeof(double)))){
     printf("malloc of S failed\n"); return(-1);
   }else{ printf("S is allocated!\n");}

   //Allocating U
   if(NULL==(U = malloc(numOfChunks*numOfChunks*n*n*sizeof(double)))){
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
   info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', numOfChunks*n, n, Rfinal, lda, S, U, numOfChunks*n, Vt, n, superb);
   printf("done!\n");

   //print_matrix_rowmajor("U", numOfChunks*n, numOfChunks*n, U, numOfChunks*n);
   //print_matrix_rowmajor("V", n, n, Vt, n);
   //print_matrix_rowmajor("S", 1, n, S, 1);

   //Now all we have to do is calculate the left eigenvectors from the left eigenvectors
   //of R and multiply against Q 


   //TODO: Write function to compare with MATLAB solutions/error checking

   //Deallocate solution matrices
   printf(" ->Deallocating malloc'ed data...");
   free(U); printf("U...");
   free(S); printf("S...");
   free(Vt); printf("Vt...");
   free(superb); printf("superb...");
   free(Rfinal); printf("Rfinal...");
   free(Qfinal); printf("Qfinal...");
   printf("done!\n");

   return(0);
}
