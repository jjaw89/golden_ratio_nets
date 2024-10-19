#!/usr/bin/env Rscript

library(dplyr)
library(tidyr)
library(xtable)

t <- read.csv("times_dem.csv") %>%
  extract(
    test,
    c("d", "n", "sequence"),
    "data_([[:digit:]]+)_([[:digit:]]+)_([[:alnum:]]+)",
    convert = TRUE
  ) %>%
  group_by(d, n, sequence, cores) %>%
  summarize(time = min(time), .groups = "drop_last") %>%
  mutate(speedup = max(time) / time) %>%
  mutate(time_speedup = sprintf("%.2f (%.1f)", time, speedup)) %>%
  pivot_wider(
    id_cols = c(d, n, sequence),
    names_from = cores,
    values_from = time_speedup
  )

rle.lengths <- rle(t$d)$lengths
first <- !duplicated(t$d)
t$d[!first] <- ""
t$d[first] <- paste0(
  "\\midrule\\multirow{",
  rle.lengths,
  "}{*}{\\textbf{",
  t$d[first],
  "}}"
)

rle.lengths <- rle(t$n)$lengths
first <- !duplicated(t$n)
t$n[!first] <- ""
t$n[first] <- paste0(
  "\\multirow{",
  rle.lengths,
  "}{*}{\\textbf{",
  t$n[first],
  "}}"
)

print(
  xtable(t),
  file = "parallel_times.tex",
  floating = FALSE,
  timestamp = NULL,
  include.rownames = FALSE,
  include.colnames = FALSE,
  booktabs = TRUE,
  sanitize.text.function = force,
  add.to.row = list(
    pos = list(
      0
    ),
    command = c(
      paste0(
        "\\multicolumn{3}{c}{} & \\multicolumn{6}{c}{\\textbf{CPU time (speedup) per number of threads}}\\\\\n",
        "\\cmidrule(l){4-9}\n",
        "N. Dim. & N. Points & Sequence & 1 & 2 & 4 & 8 & 16 & 32\\\\\n"
      )
    )
  )
)
