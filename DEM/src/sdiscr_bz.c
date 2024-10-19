#include <sdiscr/sdiscr_bz.h>
#include <sdiscr/sdiscr_log.h>
#include <sdiscr/sdiscr_utils.h>

#include <math.h>
#include <stdio.h>

#define SDISCR_BZ_INT_EPS 1e-12

#ifdef SPAM
#define SPIM
#endif
// SPIM is more benign

double globaldiscr;

size_t sdiscr_bz_intpoints(
  double** const pointset,
  const size_t dim,
  const size_t npoints,
  const double* const base
) {
  size_t n = npoints, i, j;
  for (i = 0; i < npoints; i++) {
    for (j = 0; j < dim; j++) {
      if (pointset[i][j] >= (base[j] - SDISCR_BZ_INT_EPS)) {
        n--;
        break;
      }
    }
  }
  return n;
}

// Fixes one dimension at a time, each dimension defined by a point (which
// must be on the boundary of the box).  Maintains list of points that are
// still possible (first rempoints points) and smallest base compatible with
// earlier dimension choices (boundary points must not be excluded).
double sdiscr_bz_int_poly_discr(
  double** pointset,
  const size_t dim,
  const size_t npoints,
  size_t rempoints,
  double* base,
  size_t cdim
) {
  double discr, maxdiscr = 0.0, basecopy[dim], *ptcopy[rempoints];
  int resort = 0;

  SDISCR_LOG_TRACE(
    "Dim %zu points %zu/%zu base (%g", cdim, rempoints, npoints, base[0]
  );
  for (size_t i = 1; i < dim; i++) {
    SDISCR_LOG_TRACE_APPEND(", %g", base[i]);
  }
  SDISCR_LOG_TRACE_APPEND(")\n");

  if (cdim == dim) {
    discr = 1.0;
    for (size_t i = 0; i < dim; i++) {
      discr *= base[i];
    }
    maxdiscr = fmax(
      (double)rempoints / (double)npoints - discr,
      discr - (double)sdiscr_bz_intpoints(pointset, dim, rempoints, base) /
                (double)npoints
    );

    SDISCR_LOG_TRACELN(
      "Volume %g point-share %g--%g -> discr. %g",
      discr,
      (double)sdiscr_bz_intpoints(pointset, dim, rempoints, base) /
        (double)npoints,
      (double)rempoints / (double)npoints,
      maxdiscr
    );

    if (maxdiscr > globaldiscr) {
      globaldiscr = maxdiscr;
      SDISCR_LOG_INFOLN("Improved to %g", globaldiscr);
    }
    return maxdiscr;
  }

  for (size_t i = cdim; i < dim; i++) {
    basecopy[i] = base[i];
  }

  sdiscr_utils_comparedim = cdim;
  qsort(pointset, rempoints, sizeof(double*), sdiscr_utils_cmpkeyk);

  for (size_t i = 0; i < rempoints; i++) {
    SDISCR_LOG_TRACE("Point %04zu: (%g", i, pointset[i][0]);
    for (size_t j = 1; j < dim; j++) {
      SDISCR_LOG_TRACE_APPEND(", %g", pointset[i][j]);
    }
    SDISCR_LOG_TRACE_APPEND(")\n");
  }

  for (size_t i = 0; i < rempoints; i++) {
    if (!cdim) {
      SDISCR_LOG_INFOLN("%zu/%zu", i+1, rempoints);
    }
    if (pointset[i][cdim] < base[cdim] - SDISCR_BZ_INT_EPS) {
      SDISCR_LOG_TRACELN("Skiping %zu: coord too small", i);
      continue;
    }

    base[cdim] = pointset[i][cdim];
    for (size_t j = cdim + 1; j < dim; j++) {
      if (pointset[i][j] > base[j]) {
        base[j] = pointset[i][j];
      }
    }

    if ((i && (pointset[i - 1][cdim] == pointset[i][cdim])) || (((i + 1) < rempoints) && (pointset[i + 1][cdim] == pointset[i][cdim]))) {
      resort = 1;
      for (size_t j = 0; j < rempoints; j++) {
        ptcopy[j] = pointset[j];
      }
    }

    size_t newrempoints = i + 1;
    while ((newrempoints < rempoints) &&
           (pointset[newrempoints - 1][cdim] == pointset[newrempoints][cdim])) {
      newrempoints++;
    }

    SDISCR_LOG_TRACELN("Calling %zu", i);
    discr = sdiscr_bz_int_poly_discr(
      pointset, dim, npoints, newrempoints, base, cdim + 1
    );

    if (discr > maxdiscr) {
      maxdiscr = discr;
    }

    if (resort) {
      for (size_t j = 0; j < rempoints; j++) {
        pointset[j] = ptcopy[j];
      }
      resort = 0;
    }
    for (size_t j = cdim + 1; j < dim; j++) {
      base[j] = basecopy[j];
    }
  }

  SDISCR_LOG_TRACELN("Calling 1.0");

  if (!cdim) {
    SDISCR_LOG_INFOLN("Trying 1.0");
  }

  base[cdim] = 1.0;
  discr =
    sdiscr_bz_int_poly_discr(pointset, dim, npoints, rempoints, base, cdim + 1);

  for (size_t j = cdim; j < dim; j++) {
    base[j] = basecopy[j];
  }

  if (discr > maxdiscr) {
    maxdiscr = discr;
  }

  return maxdiscr;
}

double sdiscr_bz(sdiscr_pointset* pointset) {
  double* base = malloc(pointset->d * sizeof(double));
  if (base == NULL) {
    SDISCR_LOG_ERRORLN("Failed to allocate memory for base");
    return -1;
  }
  for (size_t i = 0; i < pointset->d; i++) {
    base[i] = 0.0;
  }

  double** points = malloc(pointset->n * sizeof(double*));
  for (size_t i = 0; i < pointset->n; i++) {
    points[i] = malloc(pointset->d * sizeof(double));
    for (size_t j = 0; j < pointset->d; j++) {
      points[i][j] = pointset->data[i][j];
    }
  }

  globaldiscr = 0;
  double discr = sdiscr_bz_int_poly_discr(
    points, pointset->d, pointset->n, pointset->n, base, 0
  );
  free(base);
  return discr;
}
