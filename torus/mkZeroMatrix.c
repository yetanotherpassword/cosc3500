/* mkZeroMatrix.c  written Oct 30  by ian a. mason @ une        */
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "matrix.h"
int main(int argc, char *argv[]){
  int fd, matrix_size, row, col;
  if((argc != 3) || 
     ((fd = creat(argv[1], S_IRWXU)) == -1) ||
     ((matrix_size = atoi(argv[2])) <= 0)){
    fprintf (stderr, "Usage: %s matrix_file dimension\n", argv[0]);
    exit(1); 
  }
  for(row = 1; row <= matrix_size; row++)
    for(col = 1; col <= matrix_size; col++)
      if(set_slot(fd,matrix_size,row,col, 0) == -1){
        fprintf(stderr,"set_slot failed at [%d][%d]\n", row, col);
        exit(1); 
      }
  fprintf (stderr, "Finished writing %s\n", argv[1]);
  return 0;
}
