#ifndef SIMPLE_DISCR_H
#define SIMPLE_DISCR_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <sdiscr/sdiscr_pointset.h>

double sdiscr_simple(sdiscr_pointset *);

double rnd_coord_discr(double **pointset, int npoints, int dim, int trials);

#endif
