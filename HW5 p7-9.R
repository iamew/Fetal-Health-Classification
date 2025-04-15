data <- read.csv("hw5_missingdata.csv")
#Problem 7
complete_data <- na.omit(data)

#Problem 8
data$X1[is.na(data$X1)] <- mean(data$X1, na.rm = TRUE)
data$X2[is.na(data$X2)] <- mean(data$X2, na.rm = TRUE)
data$X3[is.na(data$X3)] <- mean(data$X3, na.rm = TRUE)


model_na <- lm(Y ~ X1 + X2 + X3, data = complete_data)
model_imput <- lm(Y ~ X1 + X2 + X3, data = data)

#Problem 9
new_obs <- data.frame(X1 = 5, X2 = -4, X3 = 1)
pred_na <- predict(model_na, newdata = new_obs)
pred_imput <- predict(model_imput, newdata = new_obs)
diff <- pred_na - pred_imput
diff