results$layer <- "GRU"
results$layer[grepl("mlp", results$model)] <- "MLP"
results$layer[grepl("rnn", results$model)] <- "RNN"
results$layer[grepl("lstm", results$model)] <- "LSTM"
results$space <- "Euclidean"
results$space[grepl("hyp", results$model)] <- "Hyperbolic"
results$test_type <- "SOTA"
results$test_type[grepl("hidden", results$model)] <- "Baseline"
results$test_type[grepl("flattened", results$model)] <- "Baseline"
results$sequence_type <- "NT"
results$sequence_type[grepl("aa", results$model)] <- "AA"


###################### BASELINE ######################

baseline_results  <- results %>% 
  dplyr::filter(test_type=="Baseline") %>% 
  group_by(model, epoch, layer, space, sequence_type) %>%
  summarise(
    ame = mean(abs(predicted_value - ground_truth)),
    sd_ame = sd(abs(predicted_value - ground_truth)),
    .groups = 'drop'
  ) 

ggplot(baseline_results, aes(x = as.numeric(epoch), y = ame, color = space, group=space)) +
  geom_line(size = 0.6, alpha = 0.8) +
  geom_ribbon(aes(ymin = ame - sd_ame, ymax = ame + sd_ame, fill = space), colour = NA, alpha = 0.15) +
  geom_line(aes(y = ame - sd_ame), alpha = 0.25) +
  geom_line(aes(y = ame + sd_ame), alpha = 0.25) +
  labs(title = "Comparison of Absolute Mean Error with Standard Deviation",
       x = "Epoch",
       y = "Absolute Mean Error") +
  scale_x_continuous(n.breaks=10) +
  facet_grid(factor(layer, levels = c("MLP", "RNN", "LSTM", "GRU")) ~ factor(sequence_type, levels = c("NT", "AA"), labels = c("Nucleotides", "Amino Acids"))) +
  theme_minimal() +
  theme(legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("analysis/plots/baseline_train.png")

###################### SOTA ######################
results$model_type <- "DSEE-GRU"
results$model_type[grepl("rnn", results$model)] <- "DSEE-RNN"
results$model_type[grepl("neuroseed", results$model)] <- "NeuroSEED"
results$embedding <- "16"
results$embedding[grepl("emb3", results$model)] <- "3"


sota_results_16  <- results %>% 
  dplyr::filter(test_type=="SOTA", embedding == "16") %>% 
  group_by(model, epoch, model_type, space) %>%
  summarise(
    ame = mean(abs(predicted_value - ground_truth)),
    sd_ame = sd(abs(predicted_value - ground_truth)),
    .groups = 'drop'
  ) 

ggplot(sota_results_16, aes(x = as.numeric(epoch), y = ame, color = space, group=space)) +
  geom_line(size = 0.6, alpha = 0.8) +
  geom_ribbon(aes(ymin = ame - sd_ame, ymax = ame + sd_ame, fill = space), colour = NA, alpha = 0.15) +
  geom_line(aes(y = ame - sd_ame), alpha = 0.25) +
  geom_line(aes(y = ame + sd_ame), alpha = 0.25) +
  labs(title = "Comparison of Absolute Mean Error with Standard Deviation",
       x = "Epoch",
       y = "Absolute Mean Error") +
  scale_x_continuous(n.breaks=10) +
  facet_wrap(~ factor(model_type, levels = c("NeuroSEED", "DSEE-RNN", "DSEE-GRU"))) +
  theme_minimal() +
  theme(legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("analysis/plots/sota_train16.png")


sota_results_3  <- results %>% 
  dplyr::filter(test_type=="SOTA", embedding == "3") %>% 
  group_by(model, epoch, model_type, space) %>%
  summarise(
    ame = mean(abs(predicted_value - ground_truth)),
    sd_ame = sd(abs(predicted_value - ground_truth)),
    .groups = 'drop'
  ) 

ggplot(sota_results_3, aes(x = as.numeric(epoch), y = ame, color = space, group=space)) +
  geom_line(size = 0.6, alpha = 0.8) +
  geom_ribbon(aes(ymin = ame - sd_ame, ymax = ame + sd_ame, fill = space), colour = NA, alpha = 0.15) +
  geom_line(aes(y = ame - sd_ame), alpha = 0.25) +
  geom_line(aes(y = ame + sd_ame), alpha = 0.25) +
  labs(title = "Comparison of Absolute Mean Error with Standard Deviation",
       x = "Epoch",
       y = "Absolute Mean Error") +
  scale_x_continuous(n.breaks=10) +
  facet_wrap(~ factor(model_type, levels = c("NeuroSEED", "DSEE-RNN", "DSEE-GRU"))) +
  theme_minimal() +
  theme(legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("analysis/plots/sota_train3.png")
