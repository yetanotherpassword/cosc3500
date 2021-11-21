/* basic matrix library for  comp309                            */
/* written by ian a. mason @ une  march 15  '99                 */            
/* touched up august '02                                        */            
/* rows & columns are numbers 1 through dimension               */
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
int get_slot(int fd, int matrix_size, int row, int col, int *slot){
  if((row <= 0) ||
     (col <= 0) ||
     (row > matrix_size) || 
     (col > matrix_size)){
    fprintf(stderr,"indexes out of range");
    return -1; 
  } else {
    off_t offset = (((row - 1)*matrix_size) + (col - 1))*sizeof(int);
    if(offset < 0){
      fprintf(stderr,"offset overflow");
      return -1; }
    else if(lseek(fd, offset, 0) < 0){
      perror("lseek failed");
      return -1; }
    else if(read(fd, slot, sizeof(int)) < 0){
      perror("read failed");
      return -1; };
    return 0;
  }
}

int set_slot(int fd, int matrix_size, int row, int col, int value){
  if((row <= 0) ||
     (col <= 0) ||
     (row > matrix_size) || 
     (col > matrix_size)){
    fprintf(stderr,"indexes out of range");
    return -1; 
  } else {
    off_t offset = (((row - 1)*matrix_size) + (col - 1))*sizeof(int);
    if(offset < 0){
      fprintf(stderr,"offset overflow");
      return -1; }
    else if(lseek(fd, offset, 0) < 0){
      perror("lseek failed");
      return -1; }
    else if(write(fd, &value, sizeof(int)) < 0){
      perror("write failed");
      return -1; };
    return 0;
  }
}

int get_row(int fd, int matrix_size, int row, int matrix_row[]){
  int column;
  if(row > matrix_size){
    fprintf(stderr,"index out of range");
    return -1; 
  } else {
    off_t offset =  ((row - 1) * matrix_size)*sizeof(int);
    if(offset < 0){
      fprintf(stderr,"offset overflow");
      return -1; }
    else if(lseek(fd, offset, 0) < 0){
      perror("lseek failed");
      return -1; }
    else for(column = 0; column < matrix_size; column++)
      if(read(fd, &matrix_row[column], sizeof(int)) < 0){
        fprintf(stderr,"read failed column = %d\n",column);
        perror("read failed");
        return -1; };
    return 0;
  }
}

int set_row(int fd, int matrix_size, int row, int matrix_row[]){
  if(row > matrix_size){
    fprintf(stderr,"indexes out of range"); 
  return -1; 
  } else {
    int column;
    off_t offset = ((row - 1) * matrix_size)*sizeof(int);
    if(offset < 0){
      fprintf(stderr,"offset overflow");
      return -1; }
    else if(lseek(fd, offset, 0) < 0){
      perror("lseek failed");
      return -1; }
    else for(column = 0; column < matrix_size; column++)
      if(write(fd, &matrix_row[column], sizeof(int)) < 0){
        perror("write failed");
        return -1; };
    return 0;
  }
}


int get_column(int fd, int matrix_size, int col, int matrix_col[]){
  int row;
  if(col > matrix_size){
    fprintf(stderr,"index out of range");
    return -1; 
  } else {
    off_t offset;
    for(row = 1; row <= matrix_size; row++){
      offset =  ((row - 1) * matrix_size + (col - 1)) * sizeof(int);
      if(offset < 0){
        fprintf(stderr,"offset overflow");
        return -1; }
      else if(lseek(fd, offset, 0) < 0){
        perror("lseek failed");
        return -1; }
      else if(read(fd, &matrix_col[row - 1], sizeof(int)) < 0){
        fprintf(stderr,"read failed row = %d\n", row);
        perror("read failed");
        return -1; };
    }
    return 0;
  }
}
