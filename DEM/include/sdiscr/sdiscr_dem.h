#ifndef SDISCR_DEM_H_INCLUDED
#define SDISCR_DEM_H_INCLUDED

#include <stddef.h>

double sdiscr_dem_int(
  double** pointset,
  size_t npoints,
  size_t dim,
  size_t rempoints,
  double** forced,
  size_t nforced,
  size_t cdim,
  double* lowerleft,
  double* upperright
);

double sdiscr_dem(double **pointset, size_t npoints, size_t dim);

#endif
