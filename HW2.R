# Load necessary library
library(survival)

# Load the dataset
data <- read.csv("hw2_braincancer.csv")

# Convert relevant columns to factors
data$stereo <- as.factor(data$stereo)
#data$status <- as.factor(data$status)

# Filter data for the SRS treatment group

head(data$status)
 
# Create a Surv object for survival analysis
surv_obj <- Surv(time = data$time, event = data$status)

# Fit the Kaplan-Meier estimator
km_fit <- survfit(surv_obj ~ data$stereo)

# Summary of Kaplan-Meier estimator
summary_km <- summary(km_fit, time = 30)
print(summary_km)

# Estimate S(30) from the Kaplan-Meier curve
# S(30) = 0.813 at closest time point to t=30 which is t=24.4

#log_rank_test <- survdiff(surv_obj ~ data$stereo)
#print(log_rank_test)

#Question 15
X = 2*(-446-(-451))
df = 2
p_value <- pchisq(X, df, lower.tail = FALSE)
p_value