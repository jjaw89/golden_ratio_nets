// expects each point on its own line with real positions.
// expects first line to be "dim npoints reals"--from thiemard

#include <stdio.h>
#include <stdlib.h>

#include "dem_discr.h"

void usage()
{
  fprintf(stderr, "Usage: dem_discr [dim npoints] [file]\n\nIf file not present, read from stdin. If dim, npoints not present, \nassume header '%%dim %%npoints reals' (e.g. '2 100 reals') in file.\n");
}

int main(int argc, char **argv)
{
  int dim, npoints,i,j;
  FILE *pointfile;
  double **pointset, upper,lower;

  FILE *random;
  unsigned int seed;
  random = fopen("/dev/random", "rb");
  fread(&seed, 4, 1, random);
  srand(seed);


  switch (argc) {
  case 0:
  case 1: // nothing, or program name only: all from stdin
    i=scanf("%d %d reals\n", &dim, &npoints);
    if (i != 2) {
      fprintf(stderr, "stdin mode and header line not present\n");
      usage();
      exit(EXIT_FAILURE);
    }
    pointfile=stdin;
    break;

  case 2: // one arg, interpret as file name
    pointfile = fopen(argv[1], "r");
    if (!pointfile) {
      fprintf(stderr, "File open failed: %s\n", argv[3]);
      exit(EXIT_FAILURE);
    }
    i=fscanf(pointfile, "%d %d reals\n", &dim, &npoints);
    if (i != 2) {
      fprintf(stderr, "stdin mode and header line not present\n");
      exit(EXIT_FAILURE);
    }
    break;

  case 3: // interpret as dim npoints
    dim=atoi(argv[1]);
    npoints=atoi(argv[2]);
    pointfile=stdin;
    break;
    
  case 4: // interpret as dim npoints file; file not allowed to have header
    dim=atoi(argv[1]);
    npoints=atoi(argv[2]);
    pointfile = fopen(argv[3], "r");
    if (!pointfile) {
      fprintf(stderr, "File open failed: %s\n", argv[3]);
      exit(EXIT_FAILURE);
    }
    break;

  default:
    usage();
    exit(EXIT_FAILURE);
  }
  
  fprintf(stderr, "Reading dim %d npoints %d\n", dim, npoints);
  pointset = malloc(npoints*sizeof(double*));
  for (i=0; i<npoints; i++) {
    pointset[i] = malloc(dim*sizeof(double));
    for (j=0; j<dim; j++) {
      // newline counts as whitespace
      if (!fscanf(pointfile, "%lg ", &(pointset[i][j]))) {
	fprintf(stderr, "File does not contain enough data points!\n");
	exit(EXIT_FAILURE);
      }

    }
  }
  fprintf(stderr, "Calling discrepancy calculation\n");
  upper = oydiscr(pointset, dim, npoints, &lower);
  printf("%g\n", upper);
  return EXIT_SUCCESS;

}

