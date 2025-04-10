# Load necessary library
library(pROC)

# Load the dataset
data <- read.csv("hw3_heart.csv")

data$heart_disease <- ifelse(data$class == 0, 0, 1)


roc_curve = roc(response = data$heart_disease, predictor = data$trestbps)

auc = auc(roc_curve)
print(auc)

#Question 13
model1 <- glm(heart_disease ~ age + thalach + chol, data = data, family = binomial)
pred1 <- predict(model1, type = "response")

roc1 <- roc(response = data$heart_disease, predictor = pred1)
auc1 <- auc(roc1)
aic1 <- AIC(model1)
print(paste("AUC for Model 1:", auc1))

model2 <- glm(heart_disease ~ age + thalach + chol + trestbps, data = data, family = binomial)

pred2 <- predict(model2, type = "response")

roc2 <- roc(response = data$heart_disease, predictor = pred2)
auc2 <- auc(roc2)
aic2 <- AIC(model2)
print(paste("AUC for Model 2:", auc2))

incremental_value <- auc2 - auc1

print(paste("Incremental Value:",incremental_value))

#Question 14
aic_diff = aic1 - aic2
print(paste("AIC Difference:", aic_diff))

#Question 15
brier_score <- function(model, data) {
  n <- nrow(data)  # Number of observations
  brier_scores <- numeric(n)
  
  for (i in 1:n) {
    # Leave-one-out cross-validation: fit model on all data except one
    train_data <- data[-i, ]
    test_data <- data[i, , drop = FALSE]
    
    # Refit the model on the training data
    model_loocv <- glm(heart_disease ~ age + sex + trestbps + chol + thalach + exang + oldpeak + slope + ca + thal + cp + fbs + restecg, data = train_data, family = binomial)
    
    # Make a prediction on the test data
    prediction <- predict(model_loocv, newdata = test_data, type = "response")

    # Calculate Brier score for this fold (predicted probability and actual outcome)
    brier_scores[i] <- (prediction - test_data$heart_disease)^2
  }
  
  # Return the average Brier score
  return(mean(brier_scores))
}

# Calculate the Brier score using LOOCV
brier_avg <- brier_score(model, data)

print(paste("Brier Score:", brier_avg))