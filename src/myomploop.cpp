#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdio.h>
using namespace std;

main()
{   clock_t  t1,t2;
    double *a, *b,*C;
    int i,j,n=800000;
    a = new double[n];
    b = new double[n];
    C = new double[n];
   t1 = clock();
    for(i=0; i<n; i++)
    {        a[i] = i;
        b[i] = i+30.4;
    }
    
    for(i=0; i<n; i++)
        C[i] = b[i] / a[i];
 
   t2 = clock();
    cout << C[4] <<  endl;
    printf("\nExecute Time = %lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));           


}
