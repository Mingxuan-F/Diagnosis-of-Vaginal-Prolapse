setwd('/Users/fanmingxuan/Desktop/GAIN-master/data')

dat <- read.csv("./imputed data1.csv", header=T)
library(glmnet)
x <- dat[,2:66]
y <- dat[,67]
x <-scale(x)

mod_cv <- cv.glmnet(x=x, y=y, family="gaussian",  intercept = F, alpha=1)
plot(mod_cv) 
best_lambda <- mod_cv$lambda.min
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)
