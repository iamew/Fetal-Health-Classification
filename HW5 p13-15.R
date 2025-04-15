#Install Packages
#install.packages("lme4", repos = "https://cloud.r-project.org/")
#install.packages("lmerTest", repos = "https://cloud.r-project.org/")
#install.packages("geepack", repos = "https://cloud.r-project.org/")

library(lme4)
library(lmerTest)
library(geepack)


data <- read.csv("hw5_aids.csv")

#Problem 13
#full_model <- lmer(CD4 ~ factor(obstime) + (1 | patient), data = data)
#reduced_model <- lmer(CD4 ~ 1 + (1 | patient), data = data)
#model_comparison <- anova(reduced_model, full_model)
#print(model_comparison)

#Problem 14
#model <- lmer(CD4 ~ obstime * AZT + (1 | patient), data = data)
#summary(model)

#Problem 15
gee_model <- geeglm(CD4 ~ obstime * AZT, id = patient, data = data, corstr = "exchangeable")
summary(gee_model)