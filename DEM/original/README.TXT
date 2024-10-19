This package contains three programs to compute the exact star
discrepancy of a set of points.  If you simply want to know the
discrepancy of a set of points, you probably want to use the third
one, which is fastest.  The other algorithms are included for
reference and comparison.

1) simple_discr: The very simple version that enumerates all (n+1)^d
   combinations of points and borders, to find the worst one.  Slow.

2) bz_discr: A slight improvement on the Bundschuh&Zhu algorithm.  The
   Bundschuh&Zhu paper claims a speedup of d! in dimension d over the
   above; I do not know whether the current improvement adds any
   further speedup in the theoretical sense.

   In either case, this algorithm is usable for low dimension (d<10,
   preferrably at most 7) and a reasonable number of points, but for
   d>10 it has very limited usability.

3) dem_discr: An implementation of the algorithm by Debkin, Eppstein,
   and Mitchell, which uses a data structure due to Overmars and Yap
   to achieve n^(d/2+1) time.  

In all cases, any errors are entirely my fault.  Feel free to contact
me about any problems. 


Instructions: on a standard unix or linux system, type "make" at a
command line to compile. Then either send the pointset via stdin:

    cat pointset | ./program_name <d> <n>

or provide a filename:

    ./program_name <d> <n> <filename>

In these cases, the input file would consist of, well, d numbers per
line and n lines, representing the coordinates.  No header.
Alternatively, a header line 

    "<d> <n> reals"

could be present in the file (as Thiemard's bracketing cover program
uses), in which case, omit <d> and <n> from the command lines above.


/Magnus Wahlström, wahl@mpi-inf.mpg.de
