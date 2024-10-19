#include <sdiscr/sdiscr_simple.h>
#include <sdiscr/sdiscr_log.h>

// change the tuple; return bool: "success"
int sdiscr_simple_step_tuple(int* tuple, size_t dim, size_t max) {
  size_t i = 0;
  while ((i < dim) && (tuple[i] + 1 == (int)max)) {
    tuple[i++] = 0;
  }
  if (i < dim) {
    tuple[i]++;
    return 1;
  } else {
    return 0;
  }
}

// pos1 is coord-wise <= pos2
int sdiscr_simple_point_in_box(double* pos1, double* pos2, size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    if (pos1[i] > pos2[i]) {
      return 0;
    }
  }
  return 1;
}

// assuming point in box, is point also on edge?
int sdiscr_simple_point_on_edge(double* pos1, double* pos2, size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    if (fabs(pos1[i] - pos2[i]) < 1e-10) {
      return 1;
    }
  }
  return 0;
}

// Returns the dis
double sdiscr_simple_box_discr(sdiscr_pointset* pointset, double* point) {
  double discr = 1.0, discr2;
  int inbox = 0, onedge = 0;
  for (size_t i = 0; i < pointset->d; i++) {
    discr *= point[i];
  }

  for (size_t i = 0; i < pointset->n; i++) {
    if (sdiscr_simple_point_in_box(pointset->data[i], point, pointset->d)) {
      inbox++;
      if (sdiscr_simple_point_on_edge(pointset->data[i], point, pointset->d)) {
        onedge++;
      }
    }
  }

  SDISCR_LOG_TRACELN("Points %d+%d vol %lg", inbox, onedge, discr);

  discr -= (double)inbox / (double)pointset->n;
  discr2 = discr + (double)onedge / (double)pointset->n;
  discr = fabs(discr);
  discr2 = fabs(discr2);

  SDISCR_LOG_TRACELN("Values %lg,%lg", discr, discr2);

  return (discr > discr2) ? discr : discr2;
}

double sdiscr_simple(sdiscr_pointset* pointset) {
  double *point, **coords, maxdiscr = 0.0, discr;
  int *tuple;
  point = malloc(pointset->d * sizeof(double));
  tuple = malloc(pointset->d * sizeof(int));
  coords = malloc(pointset->d * sizeof(double*));
  for (size_t i = 0; i < pointset->d; i++) {
    tuple[i] = 0;
    coords[i] = malloc((pointset->n + 1) * sizeof(double));
    for (size_t j = 0; j < pointset->n; j++) {
      coords[i][j] = pointset->data[j][i];
    }
    coords[i][pointset->n] = 1.0;
  }

  tuple[0] = -1;
  while (sdiscr_simple_step_tuple(tuple, pointset->d, pointset->n + 1)) {
    for (size_t i = 0; i < pointset->d; i++) {
      point[i] = coords[i][tuple[i]];
    }
    discr = sdiscr_simple_box_discr(pointset, point);
    if (discr > maxdiscr) {
      SDISCR_LOG_INFO("Point (%g", point[0]);
      for (size_t i = 1; i < pointset->d; i++) {
        SDISCR_LOG_INFO_APPEND(", %g", point[i]);
      }
      SDISCR_LOG_INFO_APPEND(") discr %g\n", discr);
      maxdiscr = discr;
    }
  }
  return maxdiscr;
}
