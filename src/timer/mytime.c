#include <stdio.h>
#include <stdlib.h>
#include "mytime.h"
#include "mach_def.h"

#if (_MACH_CLOCK == _MACH_CLOCK_GETTIME)
#if 0
struct timeval umd_base_time;

double get_seconds()
{
	struct timeval t,lapsed;
	struct timezone z;
	
	gettimeofday(&t,&z);
	if (umd_base_time.tv_usec > t.tv_usec) {
		t.tv_usec += 1000000;
		t.tv_sec--;
	}
    
	lapsed.tv_usec = t.tv_usec - umd_base_time.tv_usec;
	lapsed.tv_sec = t.tv_sec - umd_base_time.tv_sec;

	return (double)lapsed.tv_sec+((double)lapsed.tv_usec/(double)1000000.0) ;
}
#else /* 1 */
double get_seconds()
{
	struct timeval t;
	struct timezone z;
	gettimeofday(&t,&z);
	return (double)t.tv_sec+((double)t.tv_usec/(double)1000000.0);
}
#endif /* 0 */
#endif /* (_MACH_CLOCK == _MACH_CLOCK_GETTIME) */

