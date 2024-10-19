#include <sdiscr/sdiscr_dem.h>
#include <sdiscr/sdiscr_dem_parallel.h>
#include <sdiscr/sdiscr_log.h>
#include <sdiscr/sdiscr_pointset.h>
#include <sdiscr/sdiscr_utils.h>

#include <omp.h>

double sdiscr_dem_parallel_globallower;
#pragma omp threadprivate(sdiscr_dem_parallel_globallower)

int sdiscr_dem_parallel_cmpdbl(const void* a, const void* b) {
  if ((*(const double*)a) < (*(const double*)b)) {
    return -1;
  } else if ((*(const double*)a) > (*(const double*)b)) {
    return 1;
  } else {
    return 0;
  }
}

double sdiscr_dem_parallel_cell(
  const size_t npoints,
  const size_t dim,
  size_t rempoints,
  double** forced,
  const size_t nforced,
  double* lowerleft,
  double* upperright
) {
  double vol1 = 1.0;
  double vol2 = 1.0;
  for (size_t i = 0; i < dim; i++) {
    vol1 *= lowerleft[i];
    vol2 *= upperright[i];
  }

  SDISCR_LOG_TRACELN(
    "Parameters: npoints %zu, dim %zu, rempoints %zu, nforced %zu",
    npoints,
    dim,
    rempoints,
    nforced
  );
  SDISCR_LOG_TRACE("Lower: (%g", lowerleft[0]);
  for (size_t i = 1; i < dim; i++) {
    SDISCR_LOG_TRACE_APPEND(", %g", lowerleft[i]);
  }
  SDISCR_LOG_TRACE_APPEND(")\n");
  SDISCR_LOG_TRACE("Upper: (%g", upperright[0]);
  for (size_t i = 1; i < dim; i++) {
    SDISCR_LOG_TRACE_APPEND(", %g", upperright[i]);
  }
  SDISCR_LOG_TRACE_APPEND(")\n");
  SDISCR_LOG_TRACELN("Uncategorized ('forced') points are:");
  for (size_t i = 0; i < nforced; i++) {
    SDISCR_LOG_TRACE(" (%g", forced[i][0]);
    for (size_t j = 1; j < dim; j++) {
      SDISCR_LOG_TRACE_APPEND(", %g", forced[i][j]);
    }
    SDISCR_LOG_TRACE_APPEND(")\n");
  }

  double maxdiscr = vol2 - (double)rempoints / (double)npoints;
  double discr = (double)(rempoints + nforced) / (double)npoints - vol1;
  if (discr > maxdiscr) {
    maxdiscr = discr;
  }

  if (maxdiscr < sdiscr_dem_parallel_globallower) {
    return maxdiscr;
  }

  // quicker code for use in some easy cells
  // otherwise, get to work...
  size_t indexes[dim];
  double biggest[dim][nforced+1];
  double smallest[dim][nforced+1];
  for (size_t i = 0; i < dim; ++i) {
    indexes[i] = 0;
    for (size_t j = 0; j <= nforced; j++) {
      smallest[i][j] = 1.0;
      biggest[i][j] = 0.0;
    }
  }

  double coordlist[dim][nforced];
  for (size_t i = 0; i < nforced; ++i) {
    size_t status = 0;
    size_t dimension;
    size_t j = 0;
    for (; j < dim; ++j) {
      // order is chosen to handle final box
      if (forced[i][j] <= lowerleft[j]) {
        continue;
      } else if (forced[i][j] >= upperright[j]) {
        break;
      } else { // strictly internal
        if (status) {
          SDISCR_LOG_ERRORLN("Point occurs as double-internal");
          abort();
        }
        status = 1;
        dimension = j;
      }
    }
    if (j == dim) { // else: hit "break", skip
      if (status) {
        coordlist[dimension][indexes[dimension]] = forced[i][dimension];
        indexes[dimension] += 1;
      } else { // completely internal
        ++rempoints;
      }
    }
  }

  for (size_t i = 0; i < dim; ++i) {
    qsort(&(coordlist[i][0]), indexes[i], sizeof(double), sdiscr_dem_parallel_cmpdbl);
  }

  size_t maxpoints[dim];
  maxpoints[0] = indexes[0];
  for (size_t i = 1; i < dim; i++) {
    maxpoints[i] = maxpoints[i - 1] + indexes[i];
  }

  SDISCR_LOG_TRACELN(
    "Categorization: %zu lower-left, %zu internal.",
    rempoints,
    maxpoints[dim - 1]
  );

  for (size_t i = 0; i < dim; i++) {
    if (!indexes[i]) {
      SDISCR_LOG_TRACELN("Direction %zu: Nothing.", i);
      continue;
    }
    SDISCR_LOG_TRACE("Direction %zu: %g", i, coordlist[i][0]);
    for (size_t j = 1; j < indexes[i]; j++) {
      SDISCR_LOG_TRACE_APPEND(", %g", coordlist[i][j]);
    }
    SDISCR_LOG_TRACE_APPEND("\n");
  }

  // coord 0 first, since there is no recursion for that:
  smallest[0][0] = lowerleft[0];
  for (size_t j = 0; j < indexes[0]; j++) {
    smallest[0][j+1] = biggest[0][j] = coordlist[0][j];
  }
  biggest[0][indexes[0]] = upperright[0];

  SDISCR_LOG_TRACE("Direction 0 only, biggest: ");
  for (size_t j = 0; j <= maxpoints[0]; j++) {
    SDISCR_LOG_TRACE_APPEND("%g ", biggest[0][j]);
  }
  SDISCR_LOG_TRACE_APPEND("\n");
  SDISCR_LOG_TRACE("Directions 0 only, smallest: ");
  for (size_t j = 0; j <= maxpoints[0]; j++) {
    SDISCR_LOG_TRACE_APPEND("%g ", smallest[0][j]);
  }
  SDISCR_LOG_TRACE_APPEND("\n");

  for (size_t i = 1; i < dim; ++i) {
    // first the special loop for smallest: "nothing contributed by us"
    for (size_t j = 0; j <= maxpoints[i - 1]; ++j) {
      smallest[i][j] = smallest[i-1][j] * lowerleft[i];
    }
    // main loop:
    for (size_t j = 0; j < indexes[i]; j++) {
      vol1 = coordlist[i][j];
      for (size_t k = 0; k <= maxpoints[i - 1]; k++) {
        // for biggest: vol1 is coordinate that adds j new points
        vol2 = biggest[i-1][k] * vol1;
        if (vol2 > biggest[i][j+k]) {
          biggest[i][j+k] = vol2;
        }
        // for smallest: vol1 is coordinate that adds j+1 new points
        vol2 = smallest[i-1][k] * vol1;
        if (vol2 < smallest[i][j+k+1]) {
          smallest[i][j+k+1] = vol2;
        }
      }
    }
    // last, special loop for biggest: "all of us"
    for (size_t j = 0; j <= maxpoints[i - 1]; ++j) {
      vol1 = biggest[i-1][j] * upperright[i];
      if (vol1 > biggest[i][j+indexes[i]]) {
        biggest[i][j+indexes[i]] = vol1;
      }
    }

    SDISCR_LOG_TRACE("Directions 0--%zu, biggest: ", i);
    for (size_t j = 0; j <= maxpoints[i]; j++) {
      SDISCR_LOG_TRACE_APPEND("%g ", biggest[i][j]);
    }
    SDISCR_LOG_TRACE_APPEND("\n");
    SDISCR_LOG_TRACE("Directions 0--%zu, smallest: ", i);
    for (size_t j = 0; j <= maxpoints[i]; j++) {
      SDISCR_LOG_TRACE_APPEND("%g ", smallest[i][j]);
    }
    SDISCR_LOG_TRACE_APPEND("\n");
  }
  // now, use these to find lower, upper limits
  // mode: always contain "rempoints", additionally
  //         pick from smallest[dim-1], biggest[dim-1]
  maxdiscr = 0;
  for (size_t i = 0; i <= maxpoints[dim - 1]; ++i) { // i counts internal points
    // small box
    discr = (double)(rempoints + i) / (double)npoints - smallest[dim-1][i];
    if (discr > maxdiscr) {
      maxdiscr = discr;
    }
    // big box
    discr = biggest[dim-1][i] - (double)(rempoints + i) / (double)npoints;
    if (discr > maxdiscr) {
      maxdiscr = discr;
    }
  }

  if (maxdiscr > sdiscr_dem_parallel_globallower) {
    SDISCR_LOG_INFOLN("Worse bound: %g", maxdiscr);
    sdiscr_dem_parallel_globallower = maxdiscr;
  } else {
    SDISCR_LOG_TRACELN("Conclusion: %g", maxdiscr);
  }

  return maxdiscr;
}

double sdiscr_dem_serial_int(
  double** pointset,
  const size_t npoints,
  const size_t dim,
  const size_t rempoints,
  double** forced,
  const size_t nforced,
  const size_t cdim,
  double* lowerleft,
  double* upperright
) {
  if (cdim == dim) {
    return sdiscr_dem_parallel_cell(
      npoints, dim, rempoints, forced, nforced, lowerleft, upperright
    );
  }

  double** newforced = malloc((nforced + rempoints) * sizeof(double*));
  if (newforced == NULL) {
    SDISCR_LOG_ERROR("Failed to allocate memory for newforced");
    abort();
  }

  sdiscr_utils_comparedim = cdim;
  qsort(pointset, rempoints, sizeof(double*), sdiscr_utils_cmpkeyk);
  qsort(forced, nforced, sizeof(double*), sdiscr_utils_cmpkeyk);

  double lowedge = 0.0;
  double highedge = 1.0;
  double maxdiscr = 0.0;

  size_t countlimit = (size_t)sqrt((double)npoints);

  for (size_t i = 0, forcedidx = 0, previdx = 0; (i < rempoints) || (forcedidx < nforced);) {
    double coord = (i < rempoints) ? pointset[i][cdim] : 1.0;
    double forcedcoord = forcedidx < nforced ? forced[forcedidx][cdim] : 1.0;

    int wasfinal = 0;
    for (size_t newcount = 0; forcedcoord > coord && newcount <= countlimit; ++newcount) {
      ++i;
      if ((i >= rempoints) && (forcedidx >= nforced)) {
        lowerleft[cdim] = lowedge;
        highedge = upperright[cdim] = 1.0;
        wasfinal = 1;
        break;
      } else {
        coord = (i < rempoints) ? pointset[i][cdim] : 1.0;
      }
    }

    if (!wasfinal) {
      if (forcedcoord <= coord) {
        lowerleft[cdim] = lowedge;
        highedge = upperright[cdim] = forcedcoord;
      } else {
        lowerleft[cdim] = lowedge;
        highedge = upperright[cdim] = coord;
      }
    }

    if (!cdim) {
      SDISCR_LOG_TRACELN("Coord %g", highedge);
    }

    size_t curridx = i;
    size_t j = 0;
    for (; (j < nforced) && (forced[j][cdim] < highedge); j++) {
      newforced[j] = forced[j];
    }
    size_t newforcedidx = j;
    j = previdx;
    while ((j < rempoints) && (pointset[j][cdim] <= lowedge)) {
      j++;
    }
    size_t newrempoints = j;
    for (; (j < rempoints) && (pointset[j][cdim] < highedge); j++) {
      newforced[newforcedidx++] = pointset[j];
    }

    double discr;
    if (j > (curridx+1)) {
      double** newpointset = malloc(rempoints * sizeof(double));
      if (newpointset == NULL) {
        SDISCR_LOG_ERROR("Failed to allocate memory for newpointset");
        abort();
      }
      for (size_t k = 0; k < rempoints; ++k) {
        newpointset[k] = pointset[k];
      }
      discr = sdiscr_dem_serial_int(
                                    newpointset, // pointset
                                    npoints,
                                    dim,
                                    newrempoints,
                                    newforced,
                                    newforcedidx,
                                    cdim + 1,
                                    lowerleft,
                                    upperright
                                    );

      free(newpointset);
    } else {
      discr = sdiscr_dem_serial_int(
                                    pointset, // pointset
                                    npoints,
                                    dim,
                                    newrempoints,
                                    newforced,
                                    newforcedidx,
                                    cdim + 1,
                                    lowerleft,
                                    upperright
                                    );
    }

    if (discr > maxdiscr) {
      maxdiscr = discr;
    }

    while ((forcedidx < nforced) && (forced[forcedidx][cdim] == highedge)) {
      forcedidx++;
    }
    while ((i < rempoints) && (pointset[i][cdim] <= highedge)) {
      i++;
    }

    lowedge = highedge;
    previdx = i;
  }

  lowerleft[cdim] = lowedge;
  upperright[cdim] = 1.0;
        
  double discr;
  discr = sdiscr_dem_serial_int(
                                pointset,
                                npoints,
                                dim,
                                rempoints,
                                forced,
                                nforced,
                                cdim + 1,
                                lowerleft,
                                upperright
                                );

  if (discr > maxdiscr) {
    maxdiscr = discr;
  }

  free(newforced);

  return maxdiscr;
}

// "forced" points are points that are strictly between the boundaries in
// one of the cdim earlier dimensions; pointset[0--rempoints-1] are points that
// so far are at most at the lower-left corner in every dimension.
// this includes a final run with lower-left=1.
// being ON a border changes nothing:
//   ON lower-left counts as in (including when lowerleft=1)
//   ON upper-right counts as out (except if previous).
double sdiscr_dem_parallel_int(
  double** pointset,
  const size_t npoints,
  const size_t dim,
  const size_t rempoints,
  double** forced,
  const size_t nforced,
  const size_t cdim,
  double* lowerleft,
  double* upperright
) {
  if (cdim == dim) {
    return sdiscr_dem_parallel_cell(
      npoints, dim, rempoints, forced, nforced, lowerleft, upperright
    );
  }

  double complexity = pow((double)rempoints, (double)(dim - cdim) / 2 + 1);
  if (complexity < 1e8) {
    return sdiscr_dem_serial_int(
      pointset,
      npoints,
      dim,
      rempoints,
      forced,
      nforced,
      cdim,
      lowerleft,
      upperright
    );
  }

  sdiscr_utils_comparedim = cdim;
  qsort(pointset, rempoints, sizeof(double*), sdiscr_utils_cmpkeyk);
  qsort(forced, nforced, sizeof(double*), sdiscr_utils_cmpkeyk);

  double lowedge = 0.0;
  double highedge = 1.0;
  double maxdiscr = 0.0;

  size_t countlimit = (size_t)sqrt((double)npoints);
  /*size_t num_threads = (size_t)omp_get_num_threads();
  if (num_threads > 1) {
    size_t aux = rempoints / num_threads;
    size_t i = 1;
    for(; aux / i > countlimit && aux / (i+1) > 0; ++i);
    countlimit = aux / i;
  }*/

  double newlowerleft[dim];
  double newupperright[dim];
  for(size_t i = 0; i < dim; ++i) {
    newlowerleft[i] = lowerleft[i];
    newupperright[i] = upperright[i];
  }

  #pragma omp taskgroup task_reduction(max:maxdiscr)
  {
    for (size_t i = 0, forcedidx = 0, previdx = 0;
         (i < rempoints) || (forcedidx < nforced);) {
      double coord = (i < rempoints) ? pointset[i][cdim] : 1.0;
      double forcedcoord = forcedidx < nforced ? forced[forcedidx][cdim] : 1.0;

      int wasfinal = 0;
      for (size_t newcount = 0; forcedcoord > coord && newcount <= countlimit; ++newcount) {
        ++i;
        if ((i >= rempoints) && (forcedidx >= nforced)) {
          newlowerleft[cdim] = lowedge;
          highedge = newupperright[cdim] = 1.0;
          wasfinal = 1;
          break;
        } else {
          coord = (i < rempoints) ? pointset[i][cdim] : 1.0;
        }
      }

      if (!wasfinal) {
        if (forcedcoord <= coord) {
          newlowerleft[cdim] = lowedge;
          highedge = newupperright[cdim] = forcedcoord;
        } else {
          newlowerleft[cdim] = lowedge;
          highedge = newupperright[cdim] = coord;
        }
      }

      if (!cdim) {
        SDISCR_LOG_TRACELN("Coord %g", highedge);
      }

      double** newpointset = malloc(rempoints * sizeof(double));
      if (newpointset == NULL) {
        SDISCR_LOG_ERROR("Failed to allocate memory for newpointset");
        abort();
      }
      for (size_t k = 0; k < rempoints; ++k) {
        newpointset[k] = pointset[k];
      }

      double** newforced = malloc((nforced + rempoints) * sizeof(double*));
      if (newforced == NULL) {
        SDISCR_LOG_ERROR("Failed to allocate memory for newforced");
        abort();
      }
      size_t j = 0;
      for (; (j < nforced) && (forced[j][cdim] < highedge); j++) {
        newforced[j] = forced[j];
      }
      size_t newforcedidx = j;
      j = previdx;
      while ((j < rempoints) && (pointset[j][cdim] <= lowedge)) {
        j++;
      }
      size_t newrempoints = j;
      for (; (j < rempoints) && (pointset[j][cdim] < highedge); j++) {
        newforced[newforcedidx++] = pointset[j];
      }

#pragma omp task                                \
  firstprivate(newlowerleft)                   \
  firstprivate(newupperright)                  \
  firstprivate(newforcedidx)                    \
  firstprivate(newrempoints)                    \
  in_reduction(max:maxdiscr)
      {
        double discr;
        discr = sdiscr_dem_parallel_int(
                                        newpointset, // pointset
                                        npoints,
                                        dim,
                                        newrempoints,
                                        newforced,
                                        newforcedidx,
                                        cdim + 1,
                                        newlowerleft,
                                        newupperright
                                        );

        free(newpointset);
        free(newforced);

        if (discr > maxdiscr) {
          maxdiscr = discr;
        }
      }

      while ((forcedidx < nforced) && (forced[forcedidx][cdim] == highedge)) {
        forcedidx++;
      }
      while ((i < rempoints) && (pointset[i][cdim] <= highedge)) {
        i++;
      }

      lowedge = highedge;
      previdx = i;
    }

    newlowerleft[cdim] = lowedge;
    newupperright[cdim] = 1.0;
        
#pragma omp task in_reduction(max:maxdiscr)
    {
      double discr;
      discr = sdiscr_dem_parallel_int(
                                      pointset,
                                      npoints,
                                      dim,
                                      rempoints,
                                      forced,
                                      nforced,
                                      cdim + 1,
                                      newlowerleft,
                                      newupperright
                                      );

      if (discr > maxdiscr) {
        maxdiscr = discr;
      }
    }
  }

  return maxdiscr;
}

double sdiscr_dem_parallel(double** pointset, size_t npoints, size_t dim) {
  double result = -1.0;

  double* lowerleft = malloc(dim * sizeof(double));
  if (lowerleft == NULL) {
    SDISCR_LOG_ERROR("Failed to allocate memory for lowerleft");
    goto fail_lowerleft;
  }

  double* upperright = malloc(dim * sizeof(double));
  if (upperright == NULL) {
    SDISCR_LOG_ERROR("Failed to allocate memory for upperright");
    goto fail_upperright;
  }

  double** pre_force = calloc(2 * dim, sizeof(double*));
  if (pre_force == NULL) {
    SDISCR_LOG_ERROR("Failed to allocate memory for pre_force");
    goto fail_pre_force;
  }

  double** border = calloc(dim, sizeof(double*));
  if (border == NULL) {
    SDISCR_LOG_ERROR("Failed to allocate memory for border");
    goto fail_border;
  }

  for (size_t i = 0; i < dim; i++) {
    border[i] = malloc(dim * sizeof(double));
    if (border[i] == NULL) {
      SDISCR_LOG_ERROR("Failed to allocate memory for pre_force[%zu]", i);
      goto fail_pre_force_inner;
    }
    for (size_t j = 0; j < dim; j++) {
      border[i][j] = i == j ? 1.0 : 0.0;
    }
    pre_force[i] = border[i];
  }

  #pragma omp parallel
  #pragma omp single nowait
  result = sdiscr_dem_parallel_int(pointset,
                                   npoints,
                                   dim,
                                   npoints,
                                   pre_force,
                                   0,
                                   0,
                                   lowerleft,
                                   upperright
                                   );

fail_pre_force_inner:
  for (size_t i = 0; i < dim; i++) {
    free(border[i]);
  }
  free(border);
fail_border:
  free(pre_force);
fail_pre_force:
  free(upperright);
fail_upperright:
  free(lowerleft);
fail_lowerleft:

  return result;
}
