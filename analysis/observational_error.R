measurement_metrics <- function(result) {
  x = result$predicted_value
  gt = result$ground_truth
  threshold <- as.integer(mean(gt))
  similar <- gt < threshold
  dissimilar <- gt >= threshold
  predicted_similar <- x < threshold
  predicted_dissimilar <- x >= threshold
  
  TP <- sum(similar & predicted_similar)
  TN <- sum(dissimilar & predicted_dissimilar)
  FP <- sum(dissimilar & predicted_similar)
  FN <- sum(similar & predicted_dissimilar)
  
  accuracy = (TP + TN) / (TP + TN + FP + FN)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  specificity = TN / (TN + FP)
  f1_score = 2 * (precision * recall) / (precision + recall)
  return(c(Accuracy=accuracy, Precision=precision, Recall=recall, Specificity=specificity, F1_score=f1_score))
}

observational_error <- split(min_error_results, min_error_results$model) %>% 
  sapply(measurement_metrics) %>% 
  t() %>% 
  round(digits = 3) %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column(var = "Model")

write_csv(observational_error, "./observational_error.csv")
