#include <sdiscr/sdiscr_log.h>
#include <sdiscr/sdiscr_pointset.h>

#include <gsl/gsl_qrng.h>

#include <string.h>
#include <stdlib.h>

void sdiscr_pointset_free(sdiscr_pointset* const p) {
  for(size_t i = 0; i < p->n; ++i) {
    free(p->data[i]);
  }
  free(p->data);
}

sdiscr_pointset* sdiscr_pointset_alloc(const size_t npoints, const size_t dim) {
  sdiscr_pointset* p = malloc(sizeof(sdiscr_pointset));
  if (p == NULL) {
    SDISCR_LOG_ERRORLN("Failed to allocate memory for pointset");
  }

  p->n = npoints;
  p->d = dim;
  p->data = malloc(npoints * sizeof(double*));
  if (p->data == NULL) {
    SDISCR_LOG_ERRORLN("Failed to allocate memory for pointset");
    free(p);
    return NULL;
  }

  for (size_t i = 0; i < npoints; ++i) {
    p->data[i] = malloc(dim * sizeof(double));
    if (p->data[i] == NULL) {
        SDISCR_LOG_ERRORLN("Failed to allocate memory for pointset");
        for (size_t j = 0; j < i; ++j) {
          free(p->data[j]);
        }
        free(p->data);
        free(p);
        return NULL;
    }
  }

  return p;
}

sdiscr_pointset* sdiscr_pointset_from_file(FILE* const fp) {
  size_t npoints, dim;
  if (fscanf(fp, "%zu %zu", &dim, &npoints) != 2) {
    SDISCR_LOG_ERRORLN("Failed to read dimensions and number of points from file"
    );
    return NULL;
  }

  sdiscr_pointset* p = sdiscr_pointset_alloc(npoints, dim);
  if (p == NULL) {
    SDISCR_LOG_ERRORLN("Failed to allocate pointset");
    return NULL;
  }

  for (size_t i = 0; i < npoints; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      double value;
      if (fscanf(fp, "%lf", &value) != 1) {
        SDISCR_LOG_ERRORLN("Failed to read value from file");
        sdiscr_pointset_free(p);
        free(p);
        return NULL;
      }
      p->data[i][j] = value;
    }
  }

  return p;
}

// Takes into account the types and limits defined in:
// https://www.gnu.org/software/gsl/doc/html/qrng.html#c.gsl_qrng_type
gsl_qrng* new_qrng(const char* sequence, const size_t dim) {
  if (strcmp(sequence, "halton") == 0) {
    if (dim < 1 || dim > 1229) {
      SDISCR_LOG_ERRORLN(
        "Invalid n. of dimensions (%zu) for halton qrng."
        " Number of dimensions must be between 1 and 1229",
        dim
      );
      return NULL;
    }
    return gsl_qrng_alloc(gsl_qrng_halton, (unsigned int)dim);
  } else if (strcmp(sequence, "sobol") == 0) {
    if (dim < 1 || dim > 40) {
      SDISCR_LOG_ERRORLN(
        "Invalid n. of dimensions (%zu) for sobol qrng."
        " Number of dimensions must be between 1 and 40",
        dim
      );
      return NULL;
    }
    return gsl_qrng_alloc(gsl_qrng_sobol, (unsigned int)dim);
  } else if (strcmp(sequence, "niederreiter") == 0) {
    if (dim < 1 || dim > 12) {
      SDISCR_LOG_ERRORLN(
        "Invalid n. of dimensions (%zu) for niederreiter qrng."
        " Number of dimensions must be between 1 and 12",
        dim
      );
      return NULL;
    }
    return gsl_qrng_alloc(gsl_qrng_niederreiter_2, (unsigned int)dim);
  } else if (strcmp(sequence, "reversehalton") == 0) {
    if (dim < 1 || dim > 1229) {
      SDISCR_LOG_ERRORLN(
        "Invalid n. of dimensions (%zu) for reversehalton qrng."
        " Number of dimensions must be between 1 and 1229",
        dim
      );
      return NULL;
    }
    return gsl_qrng_alloc(gsl_qrng_reversehalton, (unsigned int)dim);
  } else {
    SDISCR_LOG_ERRORLN("Unknown sequence (%s)", sequence);
    return NULL;
  }
}

sdiscr_pointset* sdiscr_pointset_from_sequence(
  const char* const sequence,
  const size_t npoints,
  const size_t dim
) {
  gsl_qrng* qrng = new_qrng(sequence, dim);
  if (qrng == NULL) {
    return NULL;
  }

  sdiscr_pointset* p = sdiscr_pointset_alloc(npoints, dim);
  if (p == NULL) {
    goto freeqrng;
  }

  double* v = malloc(dim * sizeof(double));
  for (size_t i = 0; i < npoints; ++i) {
    int status = gsl_qrng_get(qrng, v);
    if (status) {
      SDISCR_LOG_ERRORLN(
        "Failed generate point with qrng: %s", gsl_strerror(status)
      );
      // free(ps) and return NULL since we don't want an incomplete pointset
      sdiscr_pointset_free(p);
      free(p);
      p = NULL;
      goto freev;
    }
    for (size_t j = 0; j < dim; ++j) {
      p->data[i][j] = v[j];
    }
  }

freev:
  free(v);
freeqrng:
  gsl_qrng_free(qrng);

  return p;
}

int sdiscr_pointset_fprintf(
  FILE* const fp,
  const sdiscr_pointset* const p,
  const char* const fmt
) {
  if (fp == NULL) {
    SDISCR_LOG_ERRORLN("NULL file pointer");
    return 1;
  }

  if (p == NULL) {
    SDISCR_LOG_ERRORLN("NULL pointset");
    return 1;
  }

  if (fprintf(fp, "%zu %zu\n", p->d, p->n) < 0) {
    SDISCR_LOG_ERRORLN("Failed to write pointset size");
    return 1;
  }

  // Nothing to print
  if (p->d == 0) {
    return 0;
  }

  for (size_t i = 0; i < p->n; ++i) {
    if (fprintf(fp, fmt, p->data[i][0]) <= 0) {
      SDISCR_LOG_ERRORLN("Failed to write pointset");
      return 1;
    }
    for (size_t j = 1; j < p->d; ++j) {
      if (fprintf(fp, " ") <= 0) {
        SDISCR_LOG_ERRORLN("Failed to write pointset");
        return 1;
      }
      if (fprintf(fp, fmt, p->data[i][j]) <= 0) {
        SDISCR_LOG_ERRORLN("Failed to write pointset");
        return 1;
      }
    }
    if (fprintf(fp, "\n") <= 0) {
      SDISCR_LOG_ERRORLN("Failed to write pointset");
      return 1;
    }
  }

  return 0;
}
