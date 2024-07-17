prediction_error <- min_error_results %>% 
  group_by(model) %>% 
  summarise(
    AME = mean(abs(predicted_value - ground_truth)),
    RMSE = sqrt(mean((predicted_value - ground_truth)^2)),
    RDE = mean(abs(predicted_value - ground_truth) / ground_truth)
  ) %>% 
  rename(Model = model)

prediction_error[,-1] <- round(prediction_error[,-1], digits = 3) 
write_csv(prediction_error, "./prediction_error.csv")
