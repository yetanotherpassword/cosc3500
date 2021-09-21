#include <omp.h>
#include <stdio.h>
#include <string.h>

//g++ -fopenmp pi.cpp

int serial_pi()
{
        static int num_steps=1048576;
        double step;
        int i; double x, pi=0.0;
	step =1.0/(double) num_steps;
        double pre=omp_get_wtime();
        {
     	        double sum=0.0; double intm=0.0;
                int tot=1;
        	int steps_per_thread = num_steps / tot; // 65536
	        //printf("id=%d steps=%d tot=%d\n",id,steps_per_thread,tot);
		for (i=0;i<num_steps;i++)
 		{
                        if (i % 65536 ==0){
                           printf("i=%d sum=%f\n",i,sum);
                           intm += sum; sum =0.0; }
			x=(i+0.5) * step;
                	sum = sum + 4.0/(1.0+x*x);
//printf("add %lf\n",4.0/(1.0+x*x));
        	}
intm+=sum;
                           printf("i=%d sum=%f\n",i,sum);
        	pi = step*intm;
//`printf("mult %f",step);
        }
        double post=omp_get_wtime();
	printf("Serial pi=%f time=%f\n",pi,post-pre);

}
int main()
{
        static int num_steps=1048576;
        double step;
        double pi=0.0;
	step =1.0/(double) num_steps;
        double pre=omp_get_wtime();
        double A[1000]={0.0};
        omp_set_num_threads(32);
        memset(A,0,1000*sizeof(double));
	#pragma omp parallel
        {
     	        double sum=0.0;
                int i;
                double x;
        	int tot = omp_get_num_threads();
        	int id = omp_get_thread_num();
        	int steps_per_thread = num_steps / tot; // 65536
	        //printf("id=%d steps=%d tot=%d\n",id,steps_per_thread,tot);
                long start=id*steps_per_thread;
                long endd=(id+1)*steps_per_thread-1;
		for (i=start;i<=endd;i++)
 		{
	           //     printf("id=%d param=%f\n",id,param);
			x=(i+0.5) * step;
                	sum = sum + 4.0/(1.0+x*x);
        	}
                printf("For id %d goes from step=%d to %d sum=%f\n",id,start,endd,sum);
        	A[id] = step*sum;
#pragma omp barrier 
;
    //    	printf("id=%d a=%f step=%d sum=%f\n",id,A[id],step,sum);
        }
        double post=omp_get_wtime();
int i;
        for (i=0;i<1000;i++)
        {
         if (A[i]==0.0) break;
          pi += (A[i]);
         printf("A[%d]=%f\n",i,A[i]);
        }
	printf("pi=%f time=%f used threads=%d\n",pi,post-pre,i);
        serial_pi();
}
