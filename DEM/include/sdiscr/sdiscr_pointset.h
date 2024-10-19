#ifndef SDISCR_POINTSET_H_INCLUDED
#define SDISCR_POINTSET_H_INCLUDED

#include <stddef.h>
#include <stdio.h>

typedef struct {
  size_t n;
  size_t d;
  double** data; 
} sdiscr_pointset;

// Frees the memory of a pointset. This does *not* free the memory of
// the pointer that is passed to this function.
void sdiscr_pointset_free(sdiscr_pointset* const p);

// Allocates memory for a pointset
sdiscr_pointset* sdiscr_pointset_alloc(const size_t ndim, const size_t npoints);

// Print the pointset to a file, according to the fmt format.
int sdiscr_pointset_fprintf(
  FILE* const fp,
  const sdiscr_pointset* const p,
  const char* const fmt
);

// Create and allocate a pointset from a file
sdiscr_pointset* sdiscr_pointset_from_file(FILE* const fp);

// Create and allocate a pointset from a quasi-random generator
// sequence
sdiscr_pointset* sdiscr_pointset_from_sequence(
  const char* sequence,
  const size_t ndim,
  const size_t npoints
);

#endif
