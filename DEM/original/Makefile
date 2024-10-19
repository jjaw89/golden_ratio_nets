CC=gcc
CFLAGS=-Wall -O2

all: bz_discr dem_discr simple_discr

bz_discr : bz_main.c bz_discr.c bz_discr.h
	$(CC) -o bz_discr bz_main.c bz_discr.c

dem_discr : dem_main.c dem_discr.c dem_discr.h bz_discr.c bz_discr.h
	$(CC) -o dem_discr dem_main.c dem_discr.c bz_discr.c -lm

simple_discr : simple_main.c simple_discr.c simple_discr.h
	$(CC) -o simple_discr simple_main.c simple_discr.c
