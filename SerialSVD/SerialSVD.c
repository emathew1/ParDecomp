#include <math.h>
#include <stdio.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include "lapacke_example_aux.h"
#include <time.h>

void tic(clock_t*);
void toc(clock_t*, clock_t*);
void parSubToc(clock_t, int, clock_t*);
void parToc(int, clock_t);
void safeMallocDouble(double **, int, char*);

int main(int argc, char *argv[])
{
   printf("\n//////////////////////////////////////////\n");
   printf("////////////SERIAL TSQR SOLVER////////////\n");
   printf("//////////////////////////////////////////\n");


   int i;

   //Number of "Simulated" Processes
   int numOfChunks = 2;

   //Read in data size file
   FILE *fid; int mnIn[2];
   fid = fopen("TestSize2.dat", "r");
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
   m = mnIn[0]; n = mnIn[1]; lda = n;
 
   //Set mSmall Size (perfectly divisible size right now)
   lapack_int mSmall = m/numOfChunks;
   
   //Allocate room for data
   double *A;   safeMallocDouble(&A, m*n, "A");
   double *R;   safeMallocDouble(&R, n*n, "R");
   double *Tau; safeMallocDouble(&Tau, n, "Tau");

   //Read in data file
   fid = fopen("TestIn2.dat", "r");
   if(fid ==NULL){
       printf("TestIn.dat file could not open!\n");
       return(-1);
   }else{
       fread(A, sizeof(double), m*n, fid);
       printf(" ->Reading in Data file of size %d by %d\n", mnIn[0], mnIn[1]);
       fclose(fid);
   }
   printf(" ->Finished reading in the %d data points!\n", m*n);

   //Timing the algorithm
   clock_t begin, end, begin2, end2, sum=0;
   tic(&begin); tic(&begin2);

   //Solving the Double GEneral matrix QR decomposition
   for(i = 0; i < numOfChunks; i++){
       printf(" ->Solving the QR decomp...%d...",i+1);
       //Need to move the pointer to the start of the next chunk after every
       //new index
       double *AchunkLocation = &A[i*mSmall*n]; 
       double *TauChunkLocation = &Tau[i*n]; 
       info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, mSmall, n, AchunkLocation, lda, TauChunkLocation);
       printf("done!\n"); 
   }



   parSubToc(begin2, numOfChunks, &sum); tic(&begin2);

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

   parSubToc(begin2, numOfChunks, &sum); tic(&begin2);

   //1st Lapack QR decomp function 
   for(i = 0; i < numOfChunks; i++){
       printf(" ->Getting the Q data from the QGEQRF output matrix...%d...", i+1);
       double *AchunkLocation = &A[i*mSmall*n];
       double *TauChunkLocation = &Tau[i*n]; 
       info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, mSmall, n, n, AchunkLocation, lda, TauChunkLocation);
       printf("done!\n");
   }
   free(Tau);

   parSubToc(begin2, numOfChunks, &sum); tic(&begin2);

   //Need to do one more QR decomp of the upper-Tri block R Matrix
   double *Rfinal; safeMallocDouble(&Rfinal, n*n, "Rfinal");
   double *Tau2;   safeMallocDouble(&Tau2, n, "Tau2");

   //2nd Lapack QR decomp function call 
   info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, numOfChunks*n, n, R, lda, Tau2);

   parSubToc(begin2, 1, &sum); tic(&begin2);

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

   parSubToc(begin2, 1, &sum); tic(&begin2);

   //Get the Q matrix back out of the last calculation
   //  For those keeping track, the Q matrices from the first QR decomp are now
   //  in A, and the Q matrix from the second decomp are now in R
   info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, numOfChunks*n, n, n, R, lda,Tau2);
   free(Tau2);

   parSubToc(begin2, 1, &sum); tic(&begin2);

   //Start getting the real Q matrix back...
   double *Qfinal; safeMallocDouble(&Qfinal, m*n, "Qfinal");
   for(i = 0; i < numOfChunks; i++){
       printf(" ->Calculating the final Q using DGEMM...%d...", i+1);
       double *AchunkLocation = &A[i*mSmall*n];
       double *RchunkLocation = &R[i*n*n];
       double *QchunkLocation = &Qfinal[i*mSmall*n];
       cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mSmall, n, n, 1.0, AchunkLocation, n, RchunkLocation, n, 0.0, QchunkLocation, n); 	
       printf("done!\n");
   }


   free(R);free(A);

   parSubToc(begin2, numOfChunks, &sum); tic(&begin2);

   //Allocating SVD Variables
   double *S;      safeMallocDouble(&S, n, "S");
   double *Utemp;  safeMallocDouble(&Utemp, n*n, "Utemp");
   double *Vt;     safeMallocDouble(&Vt, n*n, "Vt");
   double *superb; safeMallocDouble(&superb, (n-1), "superb");

   printf(" ->Calculating the SVD of Matrix R using DGESVD...");
   info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n, n, Rfinal, lda, S, Utemp, numOfChunks*n, Vt, n, superb);
   printf("done!\n");

   //Allocating U
   double *U; safeMallocDouble(&U, m*n, "U");

   parSubToc(begin2, 1, &sum); tic(&begin2);

   //Left Hand Eigenvectors of A=Q*Utemp
   for(i = 0; i < numOfChunks; i++){
       printf(" ->Calculating the Left Hand Eigenvectors of A...%d...", i+1);
       double *QfChunkLocation = &Qfinal[i*mSmall*n];
       double *UchunkLocation = &U[i*mSmall*n];
       cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mSmall, n, n, 1.0, QfChunkLocation, n, Utemp, n, 0.0, UchunkLocation, n); 	
       printf("done!\n");
   }


   //Final Time Stuff
   parSubToc(begin2, numOfChunks, &sum); 
   toc(&begin, &end);
   parToc(numOfChunks, sum);

   //print_matrix_rowmajor("U", m, n, U, n);
   //print_matrix_rowmajor("V", n, n, Vt, n);
    print_matrix_rowmajor("S", 1, n, S, 1);


   //TODO: Write function to compare with MATLAB solutions/error checking

   //Deallocate solution matrices
   printf(" ->Deallocating malloc'ed data...");
   free(Utemp);  printf("Utemp...");
   free(U);      printf("U...");
   free(S);      printf("S...");
   free(Vt);     printf("Vt...");
   free(superb); printf("superb...");
   free(Rfinal); printf("Rfinal...");
   free(Qfinal); printf("Qfinal...");
   printf("done!\n\n\n");

   return(0);
}

void safeMallocDouble(double **A, int size, char* desc){
   if(NULL==(*A = (double*)calloc(size,sizeof(double)))){
     printf(" !!!!!!!!!!!!!!!!!!!\n");
     printf(" ->%s malloc failed\n", desc);
     printf(" !!!!!!!!!!!!!!!!!!!\n");
   }
   printf(" ->%s: Memory Allocated - %lu bytes\n", desc, size*sizeof(double));
}

void tic(clock_t *begin){
   *begin = clock();
}

void toc(clock_t *begin, clock_t *end){
   *end = clock();

   printf("\n ----------------------------------\n");
   printf(" ----- tic-toc took %fs! ----\n", (double)(*end-*begin)/CLOCKS_PER_SEC);
   printf(" ----------------------------------\n\n");

}

void parSubToc(clock_t begin, int numOfChunks,  clock_t *sum){
   clock_t end = clock();
   *sum += (end - begin)/numOfChunks;
}

void parToc(int numOfChunks, clock_t sum){
   printf(" -- Estimated Parallel Algorithm --\n");
   printf(" ----------- on %d cores ----------\n", numOfChunks);
   printf(" ---- TSQR SVD took %fs! ----\n", (double)(sum)/CLOCKS_PER_SEC);
   printf(" ----- Plus Communication Time ----\n\n");
}
