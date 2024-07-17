
min_error_results$layer <- "GRU"
min_error_results$layer[grepl("mlp", min_error_results$model)] <- "MLP"
min_error_results$layer[grepl("rnn", min_error_results$model)] <- "RNN"
min_error_results$layer[grepl("lstm", min_error_results$model)] <- "LSTM"
min_error_results$space <- "Euclidean"
min_error_results$space[grepl("hyp", min_error_results$model)] <- "Hyperbolic"
min_error_results$test_type <- "SOTA"
min_error_results$test_type[grepl("hidden", min_error_results$model)] <- "Baseline"
min_error_results$test_type[grepl("flattened", min_error_results$model)] <- "Baseline"
min_error_results$sequence_type <- "NT"
min_error_results$sequence_type[grepl("aa", min_error_results$model)] <- "AA"

###################### BASELINE ######################

baseline_min_error_results  <- min_error_results %>% 
  dplyr::filter(test_type=="Baseline")


baseline_min_error_results %>% 
  mutate(error=predicted_value-ground_truth) %>% 
  ggplot(aes(x = space, y = error, fill = space)) +
  geom_boxplot() + 
  facet_grid(factor(sequence_type, levels = c("NT", "AA"), labels = c("Nucleotides", "Amino Acids")) ~ factor(layer, levels = c("MLP", "RNN", "LSTM", "GRU"))) +
  labs(title = "Prediction Error across Models for Epoch with Minimal Mean Absolute Error",
       subtitle = "CDR3 neucleotide sequences",
       y = "Error") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggsave("analysis/plots/baseline_min_error.png")


###################### SOTA ######################


min_error_results$model_type <- "DSEE-GRU"
min_error_results$model_type[grepl("rnn", min_error_results$model)] <- "DSEE-RNN"
min_error_results$model_type[grepl("neuroseed", min_error_results$model)] <- "NeuroSEED"
min_error_results$embedding <- "16"
min_error_results$embedding[grepl("emb3", min_error_results$model)] <- "3"

sota_min_error_results_16  <- min_error_results %>% 
  dplyr::filter(test_type=="SOTA", embedding == "16")


sota_min_error_results_16 %>% 
  mutate(error=predicted_value-ground_truth) %>% 
  ggplot(aes(x = space, y = error, fill = space)) +
  geom_boxplot() + 
  facet_wrap(~ factor(model_type, levels = c("NeuroSEED", "DSEE-RNN", "DSEE-GRU"))) +
  labs(title = "Prediction Error across Models for Epoch with Minimal Mean Absolute Error",
       subtitle = "CDR3 neucleotide sequences",
       y = "Error") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggsave("analysis/plots/sota_min_error16.png")



sota_min_error_results_3  <- min_error_results %>% 
  dplyr::filter(test_type=="SOTA", embedding == "3")


sota_min_error_results_3 %>% 
  mutate(error=predicted_value-ground_truth) %>% 
  ggplot(aes(x = space, y = error, fill = space)) +
  geom_boxplot() + 
  facet_wrap(~ factor(model_type, levels = c("NeuroSEED", "DSEE-RNN", "DSEE-GRU"))) +
  labs(title = "Prediction Error across Models for Epoch with Minimal Mean Absolute Error",
       subtitle = "CDR3 neucleotide sequences",
       y = "Error") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggsave("analysis/plots/sota_min_error3.png")






