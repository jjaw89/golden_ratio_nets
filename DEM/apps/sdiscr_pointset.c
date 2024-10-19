#include <sdiscr/sdiscr_log.h>
#include <sdiscr/sdiscr_pointset.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void usage(FILE* fp, const char* program) {
  fprintf(
    fp,
    "Usage:"
    " %s"
    " {DIM}"
    " {NPOINTS}"
    " {halton|sobol|niederreiter|reversehalton}"
    " [OUTPUTFILE]"
    "\n",
    program
  );
}

int main(int argc, char** argv) {
  if (argc < 4) {
    SDISCR_LOG_ERRORLN("Wrong number of arguments");
    if (argc > 0) {
      usage(stderr, argv[0]);
    } else {
      usage(stderr, "pointclu");
    }
    return 1;
  }

  const size_t ndim = (size_t)atoi(argv[1]);
  const size_t npoints = (size_t)atoi(argv[2]);
  const char* const sequence = argv[3];

  SDISCR_LOG_INFOLN(
    "Generating %zu points in %zu dimensions with sequence %s",
    npoints,
    ndim,
    sequence
  );

  sdiscr_pointset* ps = sdiscr_pointset_from_sequence(sequence, npoints, ndim);
  if (ps == NULL) {
    SDISCR_LOG_ERRORLN("Failed to open file %s", argv[4]);
    return 1;
  }

  FILE* fp;
  if (argc >= 5) {
    fp = fopen(argv[4], "w"); // Careful with dim
    if (fp == NULL) {
      SDISCR_LOG_ERRORLN("Failed to open file %s", argv[4]);
      return 1;
    }
    SDISCR_LOG_INFOLN("Writing pointset to file %s", argv[4]);
  } else {
    fp = stdout;
    SDISCR_LOG_INFOLN("Printing pointset to stdout");
  }

  int status = sdiscr_pointset_fprintf(fp, ps, "%.6lf");
  sdiscr_pointset_free(ps);
  fclose(fp);
  return status;
}
