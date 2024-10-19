#include <sdiscr/sdiscr_utils.h>
#include <stddef.h>

size_t sdiscr_utils_comparedim = 0;

int sdiscr_utils_cmpkeyk(const void *pt1, const void *pt2) {
  double a = (*(const double* const*)pt1)[sdiscr_utils_comparedim];
  double b = (*(const double* const*)pt2)[sdiscr_utils_comparedim];
  if (a<b) {
    return -1;
  } else if (a>b) {
    return 1;
  } else {
    return 0;
  }
}
