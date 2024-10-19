#ifndef SDISCR_LOG_H_INCLUDED
#define SDISCR_LOG_H_INCLUDED

#include <stdio.h>
#include <time.h>

#define SDISCR_LOG_LEVEL_ERROR_VALUE 10
#define SDISCR_LOG_LEVEL_WARN_VALUE 20
#define SDISCR_LOG_LEVEL_INFO_VALUE 30
#define SDISCR_LOG_LEVEL_DEBUG_VALUE 40
#define SDISCR_LOG_LEVEL_TRACE_VALUE 50

#ifndef SDISCR_LOG_MAX_LEVEL_VALUE
#define SDISCR_LOG_MAX_LEVEL_VALUE SDISCR_LOG_LEVEL_WARN_VALUE
#endif

#define SDISCR_LOG_ERRORLN(...)                                                \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_ERROR_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():ERROR: ", tstr, __FILE__, __LINE__, __func__    \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_ERROR(...)                                                  \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_ERROR_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():ERROR: ", tstr, __FILE__, __LINE__, __func__    \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_ERROR_APPEND(...)                                           \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_ERROR_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_WARNLN(...)                                                 \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_WARN_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {           \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():WARN: ", tstr, __FILE__, __LINE__, __func__     \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_WARN(...)                                                   \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_WARN_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {           \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():WARN: ", tstr, __FILE__, __LINE__, __func__     \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_WARN_APPEND(...)                                            \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_WARN_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {           \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_INFOLN(...)                                                 \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_INFO_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {           \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():INFO: ", tstr, __FILE__, __LINE__, __func__     \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_INFO(...)                                                   \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_INFO_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {           \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():INFO: ", tstr, __FILE__, __LINE__, __func__     \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_INFO_APPEND(...)                                            \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_INFO_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {           \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_DEBUGLN(...)                                                \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_DEBUG_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():DEBUG: ", tstr, __FILE__, __LINE__, __func__    \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_DEBUG(...)                                                  \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_DEBUG_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():DEBUG: ", tstr, __FILE__, __LINE__, __func__    \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_DEBUG_APPEND(...)                                           \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_DEBUG_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_TRACELN(...)                                                \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_TRACE_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():TRACE: ", tstr, __FILE__, __LINE__, __func__    \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_TRACE(...)                                                  \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_TRACE_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      time_t t = time(NULL);                                                   \
      char tstr[30] = { 0 };                                                   \
      strftime(tstr, sizeof(tstr), "%F %T", localtime(&t));                    \
      fprintf(                                                                 \
        stderr, "%s:%s:%d:%s():TRACE: ", tstr, __FILE__, __LINE__, __func__    \
      );                                                                       \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define SDISCR_LOG_TRACE_APPEND(...)                                           \
  do {                                                                         \
    if (SDISCR_LOG_LEVEL_TRACE_VALUE <= SDISCR_LOG_MAX_LEVEL_VALUE) {          \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#endif
