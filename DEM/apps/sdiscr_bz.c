#include <stdio.h>
#include <stdlib.h>

#include <sdiscr/sdiscr_bz.h>
#include <sdiscr/sdiscr_log.h>
#include <sdiscr/sdiscr_pointset.h>

void usage(const char* const program) {
  fprintf(
    stderr,
    "Usage: %s [FILE]"
    "\n"
    "\n"
    "If FILE not present, read pointset from stdin",
    program
  );
}

int main(int argc, char** argv) {
  FILE* pointfile = NULL;

  switch (argc) {
    case 0:
    case 1:
      pointfile = stdin;
      break;
    case 2:
      pointfile = fopen(argv[1], "r");
      if (pointfile == NULL) {
        SDISCR_LOG_ERRORLN("Error opening file \"%s\"", argv[1]);
        return EXIT_FAILURE;
      }
      break;
    default:
      if (argc == 0) {
        usage("sdiscr_bz");
      } else {
        usage(argv[0]);
      }
      return EXIT_FAILURE;
  }

  sdiscr_pointset* pointset = sdiscr_pointset_from_file(pointfile);
  fclose(pointfile);
  if (pointset == NULL) {
    if (pointfile == stdin) {
      SDISCR_LOG_ERRORLN("Error reading pointset from stdin");
    } else {
      SDISCR_LOG_ERRORLN("Error reading pointset from file \"%s\"", argv[1]);
    }
    return EXIT_FAILURE;
  }

  SDISCR_LOG_TRACELN("Calling sdiscr_simple calculation\n");
  printf("%g\n", sdiscr_bz(pointset));

  sdiscr_pointset_free(pointset);
  free(pointset);

  return EXIT_SUCCESS;
}
