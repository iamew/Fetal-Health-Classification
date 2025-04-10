# Load necessary libraries
library(pROC)

# Load the dataset (assuming it's named diabetes.csv)
diabetes_data <- read.csv("midterm_diabetes.csv")
diabetes_data <- na.omit(diabetes_data)
diabetes_data$diagnosis <- ifelse(diabetes_data$diagnosis == "Overt_Diabetic", 1, 0)
diabetes_data$diagnosis <- factor(diabetes_data$diagnosis, levels = c(0, 1))

# Fit the first logistic regression model (including relwt, instest, and glutest)
model1 <- glm(diagnosis ~ relwt + instest + glutest, data = diabetes_data, family = "binomial")

# Fit the second logistic regression model (including relwt and instest only)
model2 <- glm(diagnosis ~ relwt + instest, data = diabetes_data, family = "binomial")

# Predict probabilities for both models
pred1 <- predict(model1, type = "response")
pred2 <- predict(model2, type = "response")

# Compute ROC curves for both models
roc1 <- roc(response = diabetes_data$diagnosis, predictor = pred1)
roc2 <- roc(response = diabetes_data$diagnosis, predictor = pred2)

# Compute AUC for both models
auc1 <- auc(roc1)
auc2 <- auc(roc2)

# Compute the difference in AUC (model 1 minus model 2)
auc_diff <- auc1 - auc2

# Print the AUC difference
cat("The difference in AUC between model 1 and model 2 is:", round(auc_diff, 2), "\n")





diabetes_data <- read.csv("midterm_diabetes.csv")
diabetes_data <- na.omit(diabetes_data)
diabetes_data$diagnosis <- ifelse(diabetes_data$diagnosis == "Overt_Diabetic", 1, 0)

# Function to calculate Brier score for a given model
calculate_brier_score <- function(data, formula) {
  n <- nrow(data)
  brier_scores <- numeric(n)
  
  # Leave-One-Out Cross-Validation (LOOCV)
  for (i in 1:n) {
    train_data <- data[-i, ]  # Leave out the i-th observation
    test_data <- data[i, , drop = FALSE]
    
    # Fit the model on the training data
    fitted_model <- glm(formula, data = train_data, family = "binomial")
    
    # Make a prediction for the left-out observation
    prediction <- predict(fitted_model, newdata = test_data, type = "response")

    
    # Compute the Brier score for the left-out observation
    actual <- test_data$diagnosis
    brier_scores[i] <- (prediction - actual)^2
  }
  
  # Return the average Brier score
  return(mean(brier_scores, na.rm = TRUE))
}

# Calculate the Brier score for Model 1 (with all predictors)
brier_model1 <- calculate_brier_score(diabetes_data, diagnosis ~ relwt + instest + glutest)
cat("Brier Score for Model 1:", brier_model1, "\n")

# Calculate the Brier score for Model 2 (with only relwt and instest)
brier_model2 <- calculate_brier_score(diabetes_data, diagnosis ~ relwt + instest)
cat("Brier Score for Model 2:", brier_model2, "\n")

# Compute the difference in Brier scores (model 1 minus model 2)
brier_diff <- brier_model1 - brier_model2
cat("The difference in Brier scores between model 1 and model 2 is:", round(brier_diff, 2), "\n")


#Question 19

diabetes_data <- read.csv("midterm_diabetes.csv")
diabetes_data <- na.omit(diabetes_data)
diabetes_data$diagnosis <- ifelse(diabetes_data$diagnosis == "Overt_Diabetic", 1, 0)

# Fit the logistic regression model with the selected predictors (e.g., relwt and instest)
model <- glm(diagnosis ~ relwt + instest + glutest, data = diabetes_data, family = "binomial")

# Predict probabilities for the model
pred_prob <- predict(model, type = "response")

# Function to calculate FNR and PPV for a given threshold
calculate_fnr_ppv <- function(threshold, actual, predicted) {
  # Predicted class based on threshold
  predicted_class <- ifelse(predicted > threshold, 1, 0)
  
  # Confusion matrix components
  tp <- sum(predicted_class == 1 & actual == 1)  # True Positives
  fn <- sum(predicted_class == 0 & actual == 1)  # False Negatives
  fp <- sum(predicted_class == 1 & actual == 0)  # False Positives
  tn <- sum(predicted_class == 0 & actual == 0)  # True Negatives
  
  # Calculate FNR and PPV
  fnr <- fn / (tp + fn)  # False Negative Rate
  ppv <- tp / (tp + fp)  # Positive Predictive Value
  
  return(c(fnr, ppv))
}

# Iterate over threshold values from 0 to 1 to find the best threshold
best_threshold <- NULL
best_fnr <- 1
best_ppv <- 0
best_diff <- -Inf
for (threshold in seq(0, 1, by = 0.01)) {
  fnr_ppv <- calculate_fnr_ppv(threshold, diabetes_data$diagnosis, pred_prob)
  fnr <- fnr_ppv[1]
  ppv <- fnr_ppv[2]
  
  # We want FNR < 0.1 and maximize PPV
  if (fnr < 0.1 && ppv > best_ppv) {
    best_fnr <- fnr
    best_ppv <- ppv
    best_threshold <- threshold
  }
}

# Print the optimal threshold
best_threshold