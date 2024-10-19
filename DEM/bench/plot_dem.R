#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(tidyr)

names <- c(
  "original_dem_discr" = "Original DEM",
  "sdiscr_dem" = "SDiscr DEM",
  "sdiscr_dem_parallel" = "SDiscr DEM Parallel"
)

g <- read.csv("times_dem.csv") %>%
  extract(
    test,
    c("d", "n", "sequence"),
    "data_([[:digit:]]+)_([[:digit:]]+)_([[:alnum:]]+)",
    convert = TRUE
  ) %>%
  complete(nesting(d, n, sequence), algorithm, cores) %>%
  arrange(d, n, sequence, algorithm, cores) %>%
  fill(time) %>%
  ggplot(aes(x = cores, y = time, color = algorithm, shape = algorithm)) +
  facet_wrap(
    vars(d, n, sequence),
    scales = "free_y",
    ncol = 4,
    labeller = label_wrap_gen(multi_line=FALSE)
  ) +
  geom_point() +
  geom_line() +
  scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8, 16, 32)) +
  xlab("Number of threads") +
  ylab("Time (s)") +
  guides(
    color = guide_legend(title = "Algorithm"),
    shape = guide_legend(title = "Algorithm")
  ) +
  scale_colour_discrete(
    breaks = c("original_dem_discr", "sdiscr_dem", "sdiscr_dem_parallel"),
    labels = c("Original DEM", "SDiscr DEM", "SDiscr DEM Parallel")
  ) +
  scale_shape_discrete(
    breaks = c("original_dem_discr", "sdiscr_dem", "sdiscr_dem_parallel"),
    labels = c("Original DEM", "SDiscr DEM", "SDiscr DEM Parallel")
  )

ggsave("times_dem.pdf", g, width = 10, height = 12)

g <- read.csv("times_dem.csv") %>%
  extract(
    test,
    c("d", "n", "sequence"),
    "data_([[:digit:]]+)_([[:digit:]]+)_([[:alnum:]]+)",
    convert = TRUE
  ) %>%
  group_by(d, n, sequence, cores) %>%
  summarize(time = min(time), .groups = "drop_last") %>%
  mutate(speedup = max(time) / time, .groups = "drop") %>%
  ggplot(aes(x = cores, y = speedup)) +
  facet_wrap(
    vars(d, n, sequence),
    scales = "fixed",
    labeller = "label_both",
    ncol = 4
  ) +
  geom_line(aes(y=cores), color="blue") +
  geom_point(color = "red") +
  geom_line(color = "red") +
  scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8, 16, 32)) +
  scale_y_continuous(trans = "log2", breaks = c(1, 2, 4, 8, 16, 32))

ggsave("speedup_dem.pdf", g, width = 8, height = 12)
