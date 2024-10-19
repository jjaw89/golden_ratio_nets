// expects each point on its own line with real positions.
// expects first line to be "dim npoints reals"--from thiemard

#include <stdio.h>
#include <stdlib.h>

#include "simple_discr.h"

void usage()
{
  fprintf(stderr, "Usage: simple_discr [dim npoints] [file]\n\nIf file not present, read from stdin. If dim, npoints not present, \nassume header '%%dim %%npoints reals' (e.g. '2 100 reals') in file.\n");
}

int main(int argc, char **argv)
{
  int dim, npoints,i,j;
  FILE *pointfile;
  double **pointset;
  

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
      fscanf(pointfile, "%lg ", &(pointset[i][j]));
      // newline counts as whitespace
    }
  }
  fprintf(stderr, "Calling xdiscr calculation\n");
  printf("%g\n", exact_discr(pointset, npoints, dim));
  return EXIT_SUCCESS;

}

