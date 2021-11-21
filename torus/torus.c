#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <time.h>
#include "mmlib.h"

#define DIM 2
#define VERBOSE 0
#define TAG 0
#define VIEW_RESULT 0
int parse_args(int, char*[], int*, int*, int *, int[]);


int main(int argc, char** argv){
  int i, j, me, m, blksize, matrixDim, ntask, nproc;
  int    F[3];  /* A = F[0] B = F[1] C = F[2]               */          
  int   *M[4];  /* A = M[0] B = M[1] C = M[2]  TMP  = M[3]  */
  long startTime = time(NULL); 

  MPI_Init(&argc, &argv);
  if(parse_args(argc, argv, &m, &blksize, &ntask, F) < 0) goto exit;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  matrixDim = m * blksize;
  if(ntask != nproc){
    fprintf(stderr, "nproc = %d must equal m*m = %d\n",  nproc, m);
    goto exit;
  } else {
    MPI_Comm gridComm, rowComm, colComm;
    int sizes[DIM]    = { m, m };
    int wrap[DIM]     = { 1, 1 };
    int colDir[DIM]   = { 1, 0 };
    int rowDir[DIM]   = { 0, 1 };
    int optm = 1, gridRank, gridCoords[DIM], myRow, myCol;
    int dst, src, blksquare = blksize*blksize;
    MPI_Status status;

    if(VERBOSE || (me == 0))
      fprintf(stderr, "Creating Grid Communicator\n");


    MPI_Cart_create(MPI_COMM_WORLD, DIM, sizes, wrap, optm, &gridComm);
    MPI_Comm_rank(gridComm, &gridRank);
    MPI_Cart_coords(gridComm, gridRank, DIM, gridCoords);

    if(VERBOSE || (me == 0))
      fprintf(stderr, "Creating Column Communicator\n");

    MPI_Cart_sub(gridComm, colDir, &colComm);

    
    if(VERBOSE || (me == 0))
      fprintf(stderr, "Creating Row Communicator\n");

    MPI_Cart_sub(gridComm, rowDir, &rowComm);

    myRow = gridCoords[0];
    myCol = gridCoords[1];

    if(VERBOSE)
      fprintf(stderr, 
              "Clone %d has grid rank %d and coords [%d,%d]\n", 
              me, gridRank, gridCoords[0], gridCoords[1]);

    if(VERBOSE || (me == 0))
       fprintf(stderr, "Allocating Matrices\n");


    for(i = 0; i < 4; i++)
      M[i] = (int *)calloc(sizeof(float), blksquare);

    if (!(M[0] && M[1] && M[2] && M[3])){
      fprintf(stderr, 
              "%s: out of memory!\n", 
              argv[0]);
      for(i = 0; i < 4; i++)free(M[i]);
      goto exit;
    }
   
    if(VERBOSE || (me == 0))
      fprintf(stderr, "Initializing Matrices\n");
    
    for(j = 0; j < 2; j++)
      for(i = 0; i < blksize; i++)
        get_block_row(F[j], matrixDim, myRow, myCol, i, blksize, &(M[j][i*blksize]));
    
    
    if(VERBOSE || (me == 0))
      fprintf(stderr, "Finished Allocating and Initializing Matrices\n");
    
    src = ( myRow + 1 ) % m;
    dst = ( myRow + m - 1) % m;

    if(VERBOSE)
      fprintf(stderr, "Clone %d has src = %d and dst = %d\n", gridRank, src, dst);
 
    for (i = 0; i < m; i++) {
      int root = ( myRow  + i ) % m;
      if(VERBOSE || (me == 0))
        fprintf(stderr, "Looping: iteration  %d out of a total of %d\n", i + 1, m);
      if(root == myCol){
        //snd my A
        MPI_Bcast(M[0], blksquare, MPI_INT, root, rowComm);
        block_mult(M[2],M[0],M[1],blksize); 
      } else {
        //rcv an A
        MPI_Bcast(M[3], blksquare, MPI_INT, root, rowComm);
        block_mult(M[2],M[3],M[1],blksize); 
      }

      //rotate B's 
      MPI_Sendrecv_replace(M[1], blksquare, MPI_INT, dst, TAG, src, TAG, colComm, &status);

    }
    
    if(VERBOSE || (me == 0))
      fprintf(stderr, "Finished computation\n");

    for(i = 0; i < 2; i++)
      close(F[i]);

    if(VIEW_RESULT &&  (me == 0)){
    fprintf(stderr, "Clone %d computed:\n\n", me);
    for(i = 0; i < blksquare; i++){
      fprintf(stderr, "%5d ", (M[2][i]));
      if((( i + 1) % blksize) == 0)fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    }


    if(VERBOSE || (me == 0))
      fprintf(stderr, "Writing C\n");

    for(i = 0; i < blksize; i++)
      if(set_block_row(F[2], matrixDim, myRow, myCol, i, blksize, &(M[2][i*blksize])) != 0){
        fprintf(stderr, "set_block_row failed\n");
        goto exit;
      }
        

    if(VERBOSE || (me == 0))
      fprintf(stderr, "Cleaning up\n");

    for(i = 0; i < 4; i++)
      free(M[i]);

    close(F[2]);


  }
  if(VERBOSE || (me == 0))
    fprintf(stderr, "Exiting gracefully after %ld seconds\n", time(NULL) - startTime);
  MPI_Finalize();
  return 0;
  
 exit:
  if(VERBOSE || (me == 0))
    fprintf(stderr, "Exiting ungracefully after %ld seconds\n", time(NULL) - startTime);
  MPI_Finalize();
  return 0;
}


int parse_args(int argc, char *argv[], int *m, int *blksize, int *ntask, int F[]){
  if ((argc != 6) || 
      ((F[0] = open(argv[1], O_RDONLY)) == -1) ||
      ((F[1] = open(argv[2], O_RDONLY)) == -1) ||
      ((F[2] = open(argv[3], O_WRONLY)) == -1) ||
      ((*m   = atoi(argv[4])) <= 0)             ||
      ((*blksize   = atoi(argv[5])) <= 0)){
    fprintf(stderr, 
            "Usage: %s matrixA matrixB matrixC m blk\n", argv[0]);
    return(-1); };
  *ntask = (*m)*(*m);
  return(0); 
}


