#ifndef SDISCR_UTILS_H_INCLUDED
#define SDISCR_UTILS_H_INCLUDED

#include <stdlib.h>
#include <math.h>

extern size_t sdiscr_utils_comparedim;
#pragma omp threadprivate(sdiscr_utils_comparedim)

int sdiscr_utils_cmpkeyk(const void *pt1, const void *pt2);

#endif
