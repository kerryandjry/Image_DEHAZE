#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/time.h>		/* struct timeval */
#include <sys/resource.h>
#include <math.h>
#include <string.h>
#include "mach_def.h"

/* modified by weng */

#if (_MACH_CLOCK == _MACH_CLOCK_GETCLOCK)
struct timespec tp;
#define get_seconds()   (getclock(TIMEOFDAY, &tp), \
                        (double)tp.tv_sec + (double)tp.tv_nsec / 1000000000.0)

#define get_nseconds()  (getclock(TIMEOFDAY, &tp), \
                        (double)(1000000000*tp.tv_sec) + (double)tp.tv_nsec)
#elif (_MACH_CLOCK == _MACH_CLOCK_GETTIME)
double get_seconds();
#endif

