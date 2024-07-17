library(tidyverse)

dir_path <- "./run/results/"
file_paths <- list.files(dir_path, full.names = TRUE)

names(file_paths) <- tools::file_path_sans_ext(basename(file_paths))

results <- file_paths %>% 
  lapply(read_csv) %>% 
  lapply(mutate, epoch = gl(500, 1000)) %>% 
  bind_rows(.id = "model")


min_error_results <- results %>%
  group_by(epoch, model) %>%
  summarise(
    ame = mean(abs(predicted_value - ground_truth)),
    .groups = 'drop'
  ) %>%
  group_by(model) %>%
  slice(which.min(ame)) %>%
  ungroup() %>% 
  left_join(results) %>% 
  subset(select = -c(epoch, ame))

