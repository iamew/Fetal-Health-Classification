# Load required libraries
library(survival)
library(risksetROC)

# Load the dataset
data <- read.csv("hw4_lung.csv")
# Recode status: 1 → 0 (censored), 2 → 1 (event)
data$status <- ifelse(data$status == 1, 0, 1)

# Remove missing values for selected predictors
data <- na.omit(data[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")])

# Fit a Cox proportional hazards model
cox_model <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = data)

#Problem 1
# Compute time-dependent AUC at t = 200
#AUC_200 <- AUC.uno(Surv(data$time, data$status), Surv(data$time, data$status), predict(cox_model, type = "lp"), times = 200)

# Print the AUC value at t = 200
#round(AUC_200$auc, 2)

# Predict risk scores from Cox model
risk_scores <- predict(cox_model, type = "risk")

# Compute time-dependent ROC metrics
roc_results <- risksetROC(Stime = data$time, status = data$status,
    marker = risk_scores, 
    predict.time = 200, plot = FALSE)

# Extract PPV and NPV at threshold theta = 0.4
theta <- 0.4

predicted_event <- ifelse(risk_scores > theta, 1, 0)
observed_event <- data$status

# Calculate TP, FP, TN, FN
TP <- sum(predicted_event == 1 & observed_event == 1)
FP <- sum(predicted_event == 1 & observed_event == 0)
TN <- sum(predicted_event == 0 & observed_event == 0)
FN <- sum(predicted_event == 0 & observed_event == 1)

# Calculate PPV and NPV
PPV <- TP / (TP + FP)  # Positive Predictive Value
NPV <- TN / (TN + FN)  # Negative Predictive Value

diff <- PPV - NPV
diff